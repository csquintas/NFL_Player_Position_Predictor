import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class Evaluator:
    """Provide helpers to evaluate model predictions"""

    def compare_predictions_to_gt(
            seqs,
            preds_dict: dict[int, np.ndarray],
            game_id: int,
            play_id: int,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute frame-level errors for a single play.

        Args: 
            seqs: Iterable of player sequence dictionaries (as produced by data loaders).
            preds_dict: Mapping from nfl_id to numpy array of predicted XY points [T, 2]. 
            game_id: Identifier of the game to evaluate. 
            play_id: Identifier of the play to evaluate. 

        Returns:
            Tuple (df_frames, df_summary) where:
            - df_frames: per-frame metrics (pred/gt, dx/dy, l2)
            - df_summary: per-player RMSE over the overlapping horizon
        """
        items = [s for s in seqs if s["gid"] == game_id and s["pid"] == play_id]
        if not items:
            raise ValueError(f"No seqs for game_id={game_id}, play_id={play_id}")
        by_id = {s["nfl_id"]: s for s in items}

        rows, summary = [], []
        for nid, pred_xy in preds_dict.items():
            if nid not in by_id:
                continue
            s = by_id[nid]
            gt_xy = s.get("target_xy")
            if gt_xy is None or len(gt_xy) == 0:
                    continue

            T = min(len(pred_xy), len(gt_xy))
            pred_use = np.asarray(pred_xy[:T], dtype=float)
            gt_use   = np.asarray(gt_xy[:T], dtype=float)
            err = pred_use - gt_use                     # [T,2]
            l2 = np.sqrt(err[:, 0]**2 + err[:, 1]**2)   

            # Per-frame records
            for t in range(T):
                rows.append({
                    "game_id": game_id, "play_id": play_id, "nfl_id": nid,
                    "player_side": s.get("player_side"), "player_position": s.get("player_position"),
                    "frame": t+1,
                    "pred_x": pred_use[t, 0], "pred_y": pred_use[t, 1],
                    "gt_x":   gt_use[t, 0],   "gt_y":   gt_use[t, 1],
                    "dx": err[t, 0], "dy": err[t, 1], "L2_Error": l2[t],
                })

            rmse = float(np.sqrt(np.mean(l2**2)))
            summary.append({
                "game_id": game_id, "play_id": play_id, "nfl_id": nid,
                "player_side": s.get("player_side"), "player_position": s.get("player_position"),
                "T_overlap": int(T), "RMSE": rmse,
            })

        df_frames  = pd.DataFrame(rows).sort_values(["nfl_id", "frame"]).reset_index(drop=True) if rows else pd.DataFrame()
        df_summary = pd.DataFrame(summary).sort_values(["RMSE"]).reset_index(drop=True) if summary else pd.DataFrame()
        return df_frames, df_summary


    @staticmethod
    @torch.no_grad()
    def predict_play_trajectories(model, scaler, seqs, game_id: int, play_id: int, device: str = "cpu"):
        """
        Generate per-player future trajectories for a single play for all players in it.
        
        Args:
            model: Model
            scaler: Data scaler for feature normalization
            seqs: Iterable of player sequence dictionaries
            game_id: Game identifier 
            play_id: Play identifier
            device: Torch device

        Returns:
            dict[int, np.ndarray]: Mapping from `nfl_id` to an array shaped `[T_i, 2]`, where `T_i`
            is the number of predicted timesteps for that player. Each array contains absolute field
            coordinates measured in yards.
        
        Notes:
            B = Batch Size
            T = # of frames in batch's target (to predict)
            T_max = Max # of frames in batch's target (to predict)
        """
        hist, pad_mask, last_xy, ids, target_xy_list = EvaluatorUtils.build_play_batch_from_seqs(
            seqs, scaler, game_id, play_id
        )
        hist, pad_mask, last_xy = hist.to(device), pad_mask.to(device), last_xy.to(device)

        # Pad targets list to [B, T_max, 2] + mask
        B = len(target_xy_list)
        t_list, lengths = [], []
        for arr in target_xy_list:
            t = torch.as_tensor(arr, dtype=torch.float32, device=device)
            t_list.append(t)
            lengths.append(int(t.size(0)))
        T_max = max(lengths) if lengths else 0

        target_xy   = torch.zeros(B, T_max, 2, device=device)
        target_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        for i, t in enumerate(t_list):
            Ti = t.size(0)
            target_xy[i, :Ti] = t
            target_mask[i, :Ti] = True

        preds_full = model(hist, pad_mask, last_xy)         # [B, T_out, 2]
        preds = preds_full[:, :T_max, :]                    # [B, T_max, 2]
        pred_mask = torch.ones_like(target_mask, dtype=torch.bool)

        per_elem = (preds - target_xy).pow(2)          # [B, T_max, 2]
        per_step = per_elem.mean(dim=-1)               # [B, T_max]
        mask = target_mask & pred_mask
        mse  = per_step[mask].mean()
        rmse = mse.sqrt()
        # print(f"Masked RMSE: {rmse.item():.4f}")

        preds_np = preds.detach().cpu().numpy()
        out = {}
        for i, nid in enumerate(ids):
            Ti = lengths[i]
            out[nid] = preds_np[i, :Ti]   # trim to each player’s length
        return out

    
class EvaluatorUtils:
    """Utility methods used by the evaluator to prepare model inputs."""
    @staticmethod
    @torch.no_grad()
    def build_play_batch_from_seqs(seqs, scaler, game_id: int, play_id: int):
        """
        Build a batch of all players with data from a single play
        returns hist[B,Fmax,d], padded_frame_mask[B,Fmax], last_known_xy[B,2], ids[list], target_list (list of [T_i,2])

        Args:
            model: Model
            scaler: Data scaler for feature normalization
            seqs: Iterable of player sequence dictionaries
            game_id: Game identifier 
            play_id: Play identifier
            device: Torch device
            
        Returns:
            hist: Frame history [B,Fmax,d]
            padded_frame_mask: Which frames to use in frame history [B,Fmax]
            last_known_xy: Position right before prediction period [B,2]
            ids: Player ids [list]
            target_list: list of target frames to predict for a player (list of [T_i,2])
        
        """
        items = [s for s in seqs if s["gid"]==game_id and s["pid"]==play_id]
        if not items:
            raise ValueError(f"No seqs for game_id={game_id}, play_id={play_id}")

        # histories as torch tensors
        hists = [scaler.transform(torch.as_tensor(s["hist"], dtype=torch.float32)) for s in items]

        F_list = [h.shape[0] for h in hists]
        Fmax = max(F_list)
        d_in = hists[0].shape[1]
        B = len(hists)

        # build torch tensors directly (no numpy roundtrip)
        hist = torch.zeros(B, Fmax, d_in, dtype=torch.float32)
        padded_frame_mask = torch.ones(B, Fmax, dtype=torch.bool)  # True = padded
        last_known_xy = torch.zeros(B, 2, dtype=torch.float32)
        target_list = []
        ids = []

        for i, (s, h) in enumerate(zip(items, hists)):
            F = h.shape[0]
            hist[i, :F] = h
            padded_frame_mask[i, :F] = False

            last_xy_key = "last_known_xy" if "last_known_xy" in s else "last_xy"
            last_known_xy[i] = torch.as_tensor(s[last_xy_key], dtype=torch.float32)
            target_list.append(s["target_xy"])  
            ids.append(s["nfl_id"])

        return hist, padded_frame_mask, last_known_xy, ids, target_list