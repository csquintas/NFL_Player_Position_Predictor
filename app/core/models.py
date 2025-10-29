import torch, joblib
from src.models.PlayerTracker import PlayerTrackerTransformer
from .config import MODEL_PATH, SCALER_PATH, DEVICE

def load_model():
    """Load the PlayerTrackerTransformer checkpoint and return an eval-ready model."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = PlayerTrackerTransformer(d_in=11, T_out=100)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model

def load_scaler():
    """Load the feature scaler that accompanies the trained model."""
    return joblib.load(SCALER_PATH)
