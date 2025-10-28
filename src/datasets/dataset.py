from torch.utils.data import Dataset
import numpy as np
import torch

class Scaler:
    """Scaler (mean/std) for inputs"""

    def __init__(self):
        self.mean=None
        self.std=None
        self.eps = 1e-6

    def fit(self, X):
        """Compute feature-wise mean and standard deviation.

        Args:
            X: NumPy array `[frames, features]`.
        """
        self.mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.std  = X.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, X):
        """Normalize input data using previously computed statistics."""
        return (X - self.mean) / self.std

class NFLPredictOnlyDataset(Dataset):
    """Dataset yielding normalized historical frames and prediction targets."""

    def __init__(self, player_sequences, scaler: Scaler, fit_scaler: bool = False):
        """Store sequences and optionally fit a provided scaler.

        Args:
            player_sequences: Iterable of player sequence dictionaries.
            scaler: `Scaler` instance used to normalize histories.
            fit_scaler: When True, computes scaling stats across all histories.
        """
        self.player_sequences = player_sequences
        self.scaler = scaler
        if fit_scaler:
            X = np.concatenate([s["hist"] for s in self.player_sequences], axis=0)  # all pre-throw frames
            self.scaler.fit(X)

    def __len__(self): 
        """Return number of samples."""
        return len(self.player_sequences)

    def __getitem__(self, i):
        """Retrieve a single normalized sample for the given index."""
        s = self.player_sequences[i]
        return {
            "hist": torch.from_numpy(self.scaler.transform(s["hist"])).float(),  # [F, d_in]
            "last_known_xy": torch.from_numpy(s["last_xy"]).float(),                   # [2]
            "target_xy": torch.from_numpy(s["target_xy"]).float(),                  # [T, 2] how many frames to predict
            "gid": s["gid"], "pid": s["pid"], "nfl_id": s["nfl_id"],
        }
            

    def collate_predict_only(batch):
        """Pad variable-length histories and stack batch tensors.

        Args:
            batch: List of dataset items

        Returns:
            Tuple `(hist, padded_frame_mask, last_known_xy, target_xy, meta)` where
            tensors are padded to the longest history length in the batch.

        Notes:
            B = Batch Size
            F = # of frames in batch frame history
            F_max = Max # of frames in batch frame history
            T = # of frames in batch's target (to predict)
            T_max = Max # of frames in batch's target (to predict)
        """
        B = len(batch)
        F_max = max(b["hist"].shape[0] for b in batch)
        d_in  = batch[0]["hist"].shape[1]
        T_max = max(b["target_xy"].shape[0] for b in batch)

        hist = torch.zeros(B, F_max, d_in)
        padded_frame_mask  = torch.ones(B, F_max, dtype=torch.bool)   # True = padded
        last_known_xy   = torch.zeros(B, 2)
        target_xy = torch.zeros(B, T_max, 2, dtype=torch.float32)
        target_mask = torch.zeros(B, T_max, dtype=torch.bool)
        target_lengths = torch.zeros(B, dtype=torch.long)

        meta = {"gid": [], "pid": [], "nfl_id": []}

        for i, b in enumerate(batch):
            F = b["hist"].shape[0]
            hist[i, :F] = b["hist"]
            padded_frame_mask[i, :F]  = False
            last_known_xy[i]  = b["last_known_xy"]
            
            T_i = b["target_xy"].shape[0]
            target_xy[i, :T_i] = b["target_xy"]
            target_mask[i, :T_i] = True
            target_lengths[i] = T_i

            meta["gid"].append(b["gid"])
            meta["pid"].append(b["pid"])
            meta["nfl_id"].append(b["nfl_id"])
            meta["target_mask"] = target_mask
            meta["target_lengths"] = target_lengths

        return hist, padded_frame_mask, last_known_xy, target_xy, meta

class DatasetUtils:
    """Data Processing helper methods."""
    
    @staticmethod
    def train_test_split(player_sequences, train_split: float = 0.8, seed: int = 1):
        """Randomly split player sequences into train and test

        Args:
            player_sequences: Sequence of player dictionaries produced by `BuildDataset`.
            train_split: Fraction of samples assigned to the training partition.
            seed: RNG seed to produce deterministic splits.

        Returns:
            Tuple `(train, test)` containing lists of player sequences.
        """
        rng = np.random.default_rng(seed)            
        idx = rng.choice(len(player_sequences), size=len(player_sequences), replace=False) 
        train = [player_sequences[i] for i in idx[:int(len(player_sequences)*train_split)]]
        test = [player_sequences[i] for i in idx[int(len(player_sequences)*train_split):]]
    
        return train, test
