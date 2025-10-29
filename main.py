from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch, numpy as np

from app.core.models import load_model, load_scaler
from app.core.config import DEVICE
from app.core.plot_player_tracking import plot_trajectory
from app.schemas.prediction import PredictionRequest
from app.utils.preprocess_features import flatten_frame

app = FastAPI(title="NFL Player Trajectory Predictor")

app.mount("/images", StaticFiles(directory="images"), name="images")

model = load_model()
scaler = load_scaler()

@app.post("/predict")
def predict(req: PredictionRequest):
    hist_np = np.array([flatten_frame(f) for f in req.hist], dtype=np.float32)
    hist_np = scaler.transform(hist_np)
    hist_tensor = torch.from_numpy(hist_np).unsqueeze(0).to(DEVICE)
    mask = torch.zeros((1, hist_np.shape[0]), dtype=torch.bool).to(DEVICE)
    last_xy = torch.tensor(req.last_known_xy, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(hist_tensor, mask, last_xy)
    preds = preds[:, :req.frames_to_predict, :].squeeze(0).cpu().numpy()

    if not req.create_image:
        return {"predicted_xy": preds.tolist(), "num_frames_predicted": len(preds)}

    image_filename = plot_trajectory(req.hist, preds)

    return {
        "num_frames_predicted": len(preds),
        "predicted_xy": preds.tolist(),
        "trajectory_url": f"http://127.0.0.1:8000/images/{image_filename}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
