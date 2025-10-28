import os
import numpy as np
import pandas as pd

class Utils:
    """General utility helper functions."""

    @staticmethod
    def angle_to_sin_cos(deg: np.ndarray) -> np.ndarray:
        """
        Convert angles in degrees to their sine and cosine components.

        Args:
            deg: Array of angles expressed in degrees.

        Returns:
            NumPy array shaped like `deg` with an added trailing dimension of size two:
            `[cos(theta), sin(theta)]` for each angle.
        """
        rad = np.deg2rad(deg.astype(np.float32))
        return np.stack([np.cos(rad), np.sin(rad)], axis=-1)
