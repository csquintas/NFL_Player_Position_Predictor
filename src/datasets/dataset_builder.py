import pandas as pd
import glob
import os
from typing import Tuple, List
from tqdm.auto import tqdm
import numpy as np
from src.utils.utils import Utils
import pickle


class RawDataLoader:
    """Loading raw and processed tracking data files."""
    
    @staticmethod
    def load_raw_data_from_csv(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load input/output raw CSVs.

        Args:
            data_dir: Directory containing `input_*.csv` and `output_*.csv` files.

        Returns:
            Tuple of concatenated input and output DataFrames.
        """
        in_files = sorted(glob.glob(os.path.join(data_dir, "input_*.csv")))
        out_files = sorted(glob.glob(os.path.join(data_dir, "output_*.csv")))
        if not in_files or not out_files:
            raise FileNotFoundError("No input_*.csv / output_*.csv files found in data_dir")
        df_in  = pd.concat([pd.read_csv(f) for f in in_files], ignore_index=True)
        df_out = pd.concat([pd.read_csv(f) for f in out_files], ignore_index=True)
        
        print(f"Loaded {len(in_files)} input files and {len(out_files)} output files.")
        return df_in, df_out

    @staticmethod
    def validate_columns(df: pd.DataFrame, cols: List[str]):
        """Ensure a DataFrame contains required columns.

        Args:
            df: DataFrame to validate.
            cols: Iterable of column names expected in `df`.

        Raises:
            ValueError: If any expected column is missing.
        """
        missing_columns = [c for c in cols if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")

    @staticmethod
    def load_processed_data_from_csv(
        samples_csv: str,
        sequences_csv: str,
        T_out: int | None = None,
        only_predict: bool = True,  
        progress: bool = True,
        desc: str = "Rebuilding player sequences from CSV"
    ):
        """Rebuild player sequences from flat CSV exports with optional padding.

        Args:
            samples_csv: Path to the sample-level CSV created during export, 1 per sequence.
            sequences_csv: Path to the long-form play sequence CSV with per-frame rows.
            T_out: If given, pad/crop targets to a fixed length; otherwise keep raw lengt, used for batching.
            only_predict: Skip players who don't have target info when set to True.
            progress: Enable progress bar when True.
            desc: Optional progress bar description.

        Returns:
            List of dictionaries containing history arrays, last known positions,
            optional target arrays, and scoring masks.
        """
        df_s = pd.read_csv(samples_csv)
        df_long = pd.read_csv(sequences_csv)

        # Features to use
        HIST_FEATURES = [
            "x_pos_yds", "y_pos_yds",
            "speed_yds_sec", "accel_yds_sec2",
            "absolute_yardline",
            "ball_land_x_yds", "ball_land_y_yds",
            "player_orientation_cos_rad", "player_orientation_sin_rad",
            "player_motion_dir_cos_rad", "player_motion_dir_sin_rad",
        ]

        RawDataLoader.validate_columns(df_s, ["game_id","play_id","nfl_id","player_to_predict","has_target",
                            "num_frames_output","num_frames_hist","num_frames_target",
                            "last_x_before_pass_yds","last_y_before_pass_yds"])
        
        RawDataLoader.validate_columns(df_long, ["game_id","play_id","nfl_id","phase","frame_num","x_pos_yds","y_pos_yds",
                            "speed_yds_sec","accel_yds_sec2","absolute_yardline",
                            "ball_land_x_yds","ball_land_y_yds",
                            "player_orientation_cos_rad","player_orientation_sin_rad",
                            "player_motion_dir_cos_rad","player_motion_dir_sin_rad",
                            "is_scored_player","has_target","is_scored_frame"])

        player_sequences = []
        groups = list(df_s.groupby(["game_id","play_id","nfl_id"]))
        pbar = tqdm(total=len(groups), disable=not progress, desc=desc, leave=False, dynamic_ncols=True)

        for (gid, pid, nid), srow in groups:
            s = srow.iloc[0]
            has_target = bool(s["has_target"])
            if only_predict and not has_target:
                pbar.update(1); continue

            rows = df_long[(df_long.game_id==gid) & (df_long.play_id==pid) & (df_long.nfl_id==nid)]

            # frame history
            hist_df = rows[rows.phase=="hist"].sort_values("frame_num")
            if hist_df.shape[0] < 2:
                pbar.update(1); continue

            hist = hist_df[HIST_FEATURES].to_numpy(dtype=np.float32)                      # [F, d_in]
            last_xy = hist_df[["x_pos_yds","y_pos_yds"]].to_numpy(dtype=np.float32)[-1]

            # target data, only if the player has it
            tgt_arr, out_mask = None, None
            if has_target:
                tgt_df = rows[rows.phase=="target"].sort_values("frame_num")
                if tgt_df.shape[0] == 0:
                    pbar.update(1); continue
                tgt_xy = tgt_df[["x_pos_yds","y_pos_yds"]].to_numpy(dtype=np.float32)    # [T_raw, 2]
                mask   = tgt_df["is_scored_frame"].astype(bool).to_numpy()               # [T_raw]

                tgt_arr = tgt_xy
                out_mask = mask

            player_sequences.append(dict(
                gid=int(gid), pid=int(pid), nfl_id=int(nid),
                player_to_predict=bool(s["player_to_predict"]),
                has_target=has_target,
                num_frames_output=int(s["num_frames_output"]),
                hist=hist,                 # [F, d_in]
                last_xy=last_xy,           # [2]
                target_xy=tgt_arr,         # [T, 2] or None
                out_mask=out_mask          # [T] or None
            ))
            pbar.update(1)

        pbar.close()
        return player_sequences

    @staticmethod
    def save_sequences_to_pickle(
        player_sequences,
        pickle_path: str
    ):
        """Save pre-built player sequence.

        Args:
            player_sequences: Iterable of player sequences.
            pickle_path: Destination file path.
        """
        with open(pickle_path, "wb") as f:
            pickle.dump(player_sequences, f)
        print(f"Saved {len(player_sequences)} sequences to {pickle_path}")
    
    @staticmethod
    def load_sequences_from_pickle(
        pickle_path: str
    ):
        """Load pre-built player sequence.

        Args:
            pickle_path: Destination file path.
        
        Returns:
            player_sequences - list of dicts
        """
        with open(pickle_path, "rb") as f:
            player_sequences = pickle.load(f)
        print(f"Loaded {len(player_sequences)} sequences from {pickle_path}")

        return player_sequences
    

class BuildDataset:
    """Construct structured per-player sequences ready for torch dataset consturction."""

    @staticmethod
    def build_sequences(
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        max_out_frames: int = 100,
        export_dir: str | None = None,
        keep_unscored: bool = True,
        export_split: bool = False,
        use_target: bool = True
    ):
        """Build per-player sequences from raw data CSVs

        Args:
            df_in: DataFrame containing per-frame historical features.
            df_out: DataFrame containing target frames and scoring metadata.
            max_out_frames: Maximum number of target frames to retain per sample.
            export_dir: Optional directory to write sample/sequence CSV exports.
            keep_unscored: If False, drop players without any scored target frames.
            export_split: When exporting, additionally write split predict/non-predict CSVs.
            use_target: Use the df_out Dataframe

        Returns:
            List of dictionaries with frame history, target, and metadata.

        Notes:
        Each sample dict has:
        gid, pid, nfl_id
        player_to_predict (bool)
        has_target (bool)
        num_frames_output (int)
        player_side (str|None)   
        player_position (str|None)
        hist: [F, 11]  (pre-throw features)
        last_xy: [2]
        target_xy: [T, 2] or None
        out_mask: [T] or None
        
        B = Batch Size
        F = # of frames in batch frame history
        F_max = Max # of frames in batch frame history
        T = # of frames in batch's target (to predict)
        T_max = Max # of frames in batch's target (to predict)
        """
        out_gpn = df_out.groupby(['game_id','play_id','nfl_id']) if use_target else None

        player_sequences = []
        export_rows_samples, export_rows_long = [], []

        gby = df_in.groupby(['game_id','play_id','nfl_id'], sort=False)
        pbar = tqdm(total=gby.ngroups, desc="Building sequences", leave=False, dynamic_ncols=True)

        for (gid, pid, nid), g in gby:
            g = g.sort_values('frame_id')

            # Offense, Defense & Position
            side = str(g['player_side'].iloc[0]) if 'player_side' in g.columns else None
            pos  = str(g['player_position'].iloc[0]) if 'player_position' in g.columns else None

            # Per-frame features
            o_sc  = Utils.angle_to_sin_cos(g['o'].to_numpy())
            dir_sc= Utils.angle_to_sin_cos(g['dir'].to_numpy())
            cont = np.stack([
                g['x'].to_numpy(),
                g['y'].to_numpy(),
                g['s'].to_numpy(),
                g['a'].to_numpy(),
                g['absolute_yardline_number'].to_numpy(),
                g['ball_land_x'].to_numpy(),
                g['ball_land_y'].to_numpy(),
            ], axis=-1).astype(np.float32)

            # x,y,s,a,yardline,ballx,bally, o_cos,o_sin, dir_cos,dir_sin  => d_in = 11
            hist = np.concatenate([cont, o_sc, dir_sc], axis=-1)

            if hist.shape[0] < 2:
                pbar.update(1); continue

            last_xy = g[['x','y']].to_numpy()[-1].astype(np.float32)
            is_scored = bool(g['player_to_predict'].iloc[0]) if 'player_to_predict' in g else True

            # targets (post-throw)
            key = (gid, pid, nid)
            if use_target:
                has_target = key in out_gpn.indices
            else:
                has_target = False

            tgtT, out_mask = None, None
            nfo = 0; T = 0
            
            nfo = int(g['num_frames_output'].iloc[0])  
            
            if has_target:
                gout = out_gpn.get_group(key).sort_values('frame_id')
                tgt = gout[['x','y']].to_numpy().astype(np.float32)

                nfo = int(g['num_frames_output'].iloc[0])
                nfo = min(nfo, tgt.shape[0], max_out_frames)
                T = min(max_out_frames, tgt.shape[0])

                out_mask = np.zeros((T,), dtype=bool)
                out_mask[:nfo] = True

                if tgt.shape[0] >= T:
                    tgtT = tgt[:T]
                else:
                    pad = np.repeat(tgt[-1][None, :], T - tgt.shape[0], axis=0)
                    tgtT = np.concatenate([tgt, pad], axis=0)
            else:
                if not keep_unscored:
                    pbar.update(1); continue

            sample = dict(
                gid=int(gid), pid=int(pid), nfl_id=int(nid),
                player_to_predict=is_scored,
                has_target=has_target,
                num_frames_output=int(nfo),
                player_side=side,          
                player_position=pos,          
                hist=hist,                     # [F, 11]
                last_xy=last_xy,               # [2]
                target_xy=tgtT,                # None if no targets
                out_mask=out_mask              # None if no targets
            )

            player_sequences.append(sample)

            # export rows
            if export_dir is not None:
                F = hist.shape[0]
                export_rows_samples.append({
                    'game_id': int(gid),
                    'play_id': int(pid),
                    'nfl_id': int(nid),
                    'player_to_predict': is_scored,
                    'has_target': has_target,
                    'num_frames_output': int(nfo),
                    'num_frames_hist': int(F),
                    'num_frames_target': int(T),
                    'last_x_before_pass_yds': float(last_xy[0]),
                    'last_y_before_pass_yds': float(last_xy[1]),
                    'player_side': side,        
                    'player_position': pos,    
                })
                # history rows
                for t in range(F):
                    export_rows_long.append({
                        'game_id': int(gid), 'play_id': int(pid), 'nfl_id': int(nid),
                        'phase': 'hist', 'frame_num': t+1,
                        'x_pos_yds': float(hist[t, 0]),
                        'y_pos_yds': float(hist[t, 1]),
                        'speed_yds_sec': float(hist[t, 2]),
                        'accel_yds_sec2': float(hist[t, 3]),
                        'absolute_yardline': float(hist[t, 4]),
                        'ball_land_x_yds': float(hist[t, 5]),
                        'ball_land_y_yds': float(hist[t, 6]),
                        'player_orientation_cos_rad': float(hist[t, 7]),
                        'player_orientation_sin_rad': float(hist[t, 8]),
                        'player_motion_dir_cos_rad': float(hist[t, 9]),
                        'player_motion_dir_sin_rad': float(hist[t,10]),
                        'is_scored_player': is_scored,
                        'has_target': has_target,
                        'is_scored_frame': None,    # N/A for hist
                        'player_side': side,       
                        'player_position': pos,    
                    })
                # target rows
                if has_target:
                    for u in range(T):
                        export_rows_long.append({
                            'game_id': int(gid), 'play_id': int(pid), 'nfl_id': int(nid),
                            'phase': 'target', 'frame_num': u+1,
                            'x_pos_yds': float(tgtT[u, 0]),
                            'y_pos_yds': float(tgtT[u, 1]),
                            'speed_yds_sec': None,
                            'accel_yds_sec2': None,
                            'absolute_yardline': None,
                            'ball_land_x_yds': None,
                            'ball_land_y_yds': None,
                            'player_orientation_cos_rad': None,
                            'player_orientation_sin_rad': None,
                            'player_motion_dir_cos_rad': None,
                            'player_motion_dir_sin_rad': None,
                            'is_scored_player': is_scored,
                            'has_target': True,
                            'is_scored_frame': bool(out_mask[u]),
                            'player_side': side,        
                            'player_position': pos, 
                        })
            pbar.update(1)

        pbar.close()


        if export_dir is not None:
            os.makedirs(export_dir, exist_ok=True)
            df_samples = pd.DataFrame(export_rows_samples).drop_duplicates()
            df_long    = pd.DataFrame(export_rows_long)

            df_samples.to_csv(os.path.join(export_dir, "samples.csv"), index=False)
            df_long.to_csv(os.path.join(export_dir, "all_sequences.csv"), index=False)

            if export_split:
                df_s_pred  = df_samples[df_samples['has_target']]
                df_s_npred = df_samples[~df_samples['has_target']]
                df_l_pred  = df_long[df_long['has_target']]
                df_l_npred = df_long[~df_long['has_target']]

                df_s_pred.to_csv(os.path.join(export_dir, "samples_predict.csv"), index=False)
                df_s_npred.to_csv(os.path.join(export_dir, "samples_non_predict.csv"), index=False)
                df_l_pred.to_csv(os.path.join(export_dir, "sequences_predict.csv"), index=False)
                df_l_npred.to_csv(os.path.join(export_dir, "sequences_non_predict.csv"), index=False)
            
        return player_sequences
