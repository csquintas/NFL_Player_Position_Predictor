from pydantic import BaseModel
from typing import List

class Frame(BaseModel):
    position: List[float]
    velocity: float
    accel: float
    player_orientation: float
    player_motion_direction: float
    absolute_yardline: float
    ball_land_position: List[float]


class PredictionRequest(BaseModel):
    hist: List[Frame]
    last_known_xy: List[float] # used for output delta
    frames_to_predict: int
    create_image: bool