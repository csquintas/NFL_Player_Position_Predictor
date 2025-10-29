import numpy as np
from src.utils.utils import Utils

def flatten_frame(frame):
    """
    Create model ready features from player
    """
    orientation = Utils.angle_to_sin_cos(np.array([frame.player_orientation])).ravel()
    motion_dir = Utils.angle_to_sin_cos(np.array([frame.player_motion_direction])).ravel()

    return [
        frame.position[0], frame.position[1],
        frame.velocity, frame.accel, frame.absolute_yardline,
        frame.ball_land_position[0], frame.ball_land_position[1],
        orientation[0], orientation[1], motion_dir[0], motion_dir[1],
    ]