import os, uuid
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from .config import IMAGES_DIR

def plot_trajectory(hist, preds):
    """Plot the observed history and predicted trajectory of a single player.

    Args:
        hist: Iterable of frames whose `position` attribute is an (x, y) pair.
        preds: `numpy.ndarray` of shape (N, 2) containing predicted x/y positions.

    Returns:
        Filename of the saved plot within `IMAGES_DIR`.
    """
    plt.figure(figsize=(8, 4))
    hist_xy = np.array([f.position for f in hist], dtype=float)
    plt.plot(hist_xy[:, 0], hist_xy[:, 1], color="blue", linestyle="-", linewidth=1.8, label="Pre-Throw")
    plt.scatter(hist_xy[-1, 0], hist_xy[-1, 1], color="blue", marker="x", s=40, label="Release Point")
    plt.plot(preds[:, 0], preds[:, 1], color="green", linestyle=":", marker="s", markersize=3, label="Predicted Post-Throw")
    plt.title("Predicted Player Trajectory")
    plt.xlabel("X (yards)")
    plt.ylabel("Y (yards)")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    filename = f"api_plot_{uuid.uuid4().hex}.png"
    save_path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return filename
