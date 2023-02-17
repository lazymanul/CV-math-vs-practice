import os 
import sys

# add parent dir to import path 
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

import pytest

import numpy as np
from utils.transform3d import Rotate, Translate
import utils.camera_adv as ca


def test_PerspectiveCamera_init():
    cam = ca.PerspectiveCameras()
    proj_tansform = cam.get_projection_transform()
    K = np.array([[1,    0,    0,   0],
                  [0,    1,    0,   0],
                  [0,    0,    1,   0],
                  [0,    0,    0,   1]], dtype=np.float32)
    assert np.allclose(K, proj_tansform.get_matrix())


def test_PerspectiveCamera_project_points():
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    T = np.array([-5,-6,-7])
    cam = ca.PerspectiveCameras(R=R, T=T)

    points = np.array([[6,8,10]])
    points_proj = cam.transform_points(points)
    points_cam_space = np.array([[1,2,3]])
    assert np.allclose(points_cam_space, points_proj)    
    
    points_cam_frame = np.array([[1,2,3]])
    points_unproj_cam = cam.unproject_points(points_proj, world_coordinates=False)
    assert np.allclose(points_cam_frame, points_unproj_cam)

    points_unproj_world = cam.unproject_points(points_proj)
    assert np.allclose(points, points_unproj_world)    


def test_PerspectiveCamera_project_points_2():
    # camera pose in world frame
    R_i = np.array([[1,  0,  0],
                    [0,  0, -1],
                    [0,  1,  0]])
    R_k = np.array([[ np.cos(np.pi/4),  np.sin(np.pi/4),  0],
                    [-np.sin(np.pi/4),  np.cos(np.pi/4),  0],
                    [           0,              0,        1]])
    O_c = np.array([5,6,7])
    camera_to_world_transform = Rotate(R_i).compose(Rotate(R_k)).compose(Translate(O_c))
        
    world_to_camera_matrix = camera_to_world_transform.inverse().get_matrix()
    R = world_to_camera_matrix[:3, :3]
    T = world_to_camera_matrix[:3, 3]
    cam = ca.PerspectiveCameras(R=R, T=T)

    points = np.array([[1,2,3]])
    points_proj = cam.transform_points(points)
    points_unproj = cam.unproject_points(points_proj)
    assert np.allclose(points, points_unproj)


def test_get_world_to_view_transform():
    T = np.array([1, 2, 3])
    c = np.cos(np.pi / 3)
    s = np.sin(np.pi / 3)
    R = np.array([[1, 0, 0],
                  [0, c,-s],
                  [0, s, c]])

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = T

    transform = ca.get_world_to_view_transform(R, T)
    assert np.allclose(Rt, transform.get_matrix())

