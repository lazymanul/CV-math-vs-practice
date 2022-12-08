import os 
import sys

# add parent dir to import path 
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

import pytest

import numpy as np
from utils.camera import Camera

tolerance = 10e-6

def test_look_at_default():
    cam = Camera.from_look_at()
    R_test = np.array([[ 0.70710678, -0.70710678,  0.0       ],
                       [-0.40824829, -0.40824829,  0.81649658],
                       [-0.57735027, -0.57735027, -0.57735027]])
    t_test = np.array([0.0, 0.0, 3.46410162])

    assert np.linalg.det(cam.R) - 1 < tolerance
    assert np.allclose(cam.R, R_test, rtol=tolerance)
    assert np.allclose(cam.t, t_test, rtol=tolerance)
    