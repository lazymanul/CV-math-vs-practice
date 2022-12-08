from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

from scipy.linalg import expm # type: ignore

class Camera():

    @classmethod 
    def from_look_at(cls, 
                     cam_position: NDArray = np.array([2, 2, 2]), 
                     cam_target: NDArray = np.array([0, 0, 0]),
                     up: NDArray = np.array([0, 0, 1])) -> Camera:

        if cam_position is not None:            
            cam_direction = (cam_target - cam_position) / np.linalg.norm(cam_target - cam_position)
            cam_x_axis = np.cross(up, cam_direction) / np.linalg.norm(np.cross(up, cam_direction))
            cam_up = np.cross(cam_direction, cam_x_axis)

            # R and t tensors are computed in camera frame
            R_cam = np.vstack((cam_x_axis, cam_up, cam_direction))
            t_cam = - R_cam @ np.expand_dims(cam_position, axis=1)
            Rt = np.hstack((R_cam, t_cam))
            K = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
                          
            return cls(K, Rt)
        else:
            raise ValueError("P matrix is not defined")


    @classmethod 
    def from_projection_matrix(cls, P: Optional[NDArray] = None) -> Camera:

        if P is not None:
            K, R, t = cls.factor_camera_matrix(P)
            Rt = np.hstack((R, t))
            return cls(K, Rt)
        else:
            raise ValueError("P matrix is not defined")


    def __init__(self, K: Optional[NDArray], Rt: Optional[NDArray]):

        if K is not None:            
            if Rt is not None:
                self.P: NDArray = K @ Rt       # projection matrix
                self.K: NDArray = K            # intrinsic parameters matrix
                self.R: NDArray = Rt[:,:3]     # rotation matrix
                self.t: NDArray = Rt[:,3]      # translation matrix
                self.c: NDArray = -self.R.T @ self.t
            else:
                raise ValueError("Rt matrix is not defined")
        else:
            raise ValueError("K matrix is not defined")


    def project(self, X: NDArray, clip_plane: bool = False) -> NDArray:

        points = self.P @ X # project
        
        # remove points behind the camera image plane
        if clip_plane:        
            cam_plane_z = -max(self.K[0,0], self.K[1,1])
            z_filter = points.T[:, 2] < cam_plane_z
            points = points.T[z_filter].T

        points = points / points[2]   # normalize
        return points


    @staticmethod
    def rotation_matrix(a: NDArray) -> NDArray:

        R = np.eye(4)                
        R[:3, :3] = expm([[    0, -a[2],  a[1]],
                          [ a[2],     0, -a[0]],
                          [-a[0],  a[0],     0]])
        return R

    @staticmethod
    def factor_camera_matrix(P) -> Tuple[NDArray, NDArray, NDArray]:

        R, K = np.linalg.qr(P[:,:3])
        # additional transfor for rotation sign compensation
        # M = Q T T^(-1) R  + ambiguos notation about R...
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1,1] *= -1

        K = K @ T
        R = T @ R # T^(-1) = T

        t = np.linalg.inv(K) @ P[:,3]

        return K, R, t


    def get_camera_center(self) -> NDArray:
        return self.c


