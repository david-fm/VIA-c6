import json
import numpy as np

def read_json_calib(calib_file: str) -> dict:
    """
    Read a json calibration file and return the calibration dictionary.
    """
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    return calib

def k_matrix(calib: dict) -> np.ndarray:
    """
    Return the camera matrix as a numpy array.
    """
    return np.array(calib['camera_matrix']).reshape(3, 3)

def d_coeff(calib: dict) -> np.ndarray:
    """
    Return the distortion coefficients as a numpy array.
    """
    return np.array(calib['distortion_coefficients'])
