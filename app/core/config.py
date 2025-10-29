import torch

MODEL_PATH = "model_registry/best_model_train.pt"
SCALER_PATH = "model_registry/scaler.pkl"
DEVICE = "mps" if torch.mps.is_available() else "cpu"
IMAGES_DIR = "images"