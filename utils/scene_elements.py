from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


class Basis:

    def __init__(self,                  
                 name: str = 'W',                 
                 origin: NDArray = np.zeros(3),
                 basis: NDArray = np.eye(3),
                 parent_basis: Optional[NDArray] = None):
        self.name = name
        self.origin = origin
        self.basis = basis
        self.parent_basis = parent_basis
        self.child_bases = [] 

    def draw(self, ax: plt.Axes) -> None:
        ax.quiver(*self.origin, *self.basis[:,0], color='r')
        ax.quiver(*self.origin, *self.basis[:,1], color='g')
        ax.quiver(*self.origin, *self.basis[:,2], color='b')

    def get_transform(self) -> NDArray:
        """
        Computes basis transform matrix

        Computes transformation matrix from parent basis to current one.
        If parent basis is not defined, world basis is used.

        Returns:        
        (4,4) numpy array: Transformation matrix
        """
        T = np.zeros((4,4))
        T[:3,:3] = self.basis
        T[:3,3] = self.origin
        T[3,3] = 1
        if self.parent_basis is None:            
            return T
        else:
            return T @ self.parent_basis.get_transform() # TO-DO make tests


class Point:

    def __init__(self,                  
                 name: str = 'P',
                 coords: NDArray = np.ones(3),
                 basis: Optional[NDArray] = None):
        self.name = name
        self.coords = coords
        self.basis = basis

    def draw(self, ax: plt.Axes) -> None:
        if self.basis is None:
            raise AttributeError('point doesn\'t have a basis')
        else:            
            T = self.basis.get_transform()            
            hom_coords = np.hstack((self.coords, [1]))
            world_coords = (T @ hom_coords)[:3]
            ax.plot(*world_coords, marker='o', color='black', markersize=5)


class Camera():
    """Camera directed along z-axis"""

    def __init__(self, 
                 basis: Basis = Basis(),
                 K: NDArray = np.eye(3)):        
        self.basis = basis
        T = self.basis.get_transform()
        self.Rt = np.linalg.inv(T)[:3,:]        
        self.K = K
        self.P = self.K @ self.Rt
    
    def project(self, points: NDArray, clip_plane: bool = True) -> NDArray:  
        hom_points = np.vstack((points.T, np.ones(points.shape[0])))
        camera_points = self.P @ hom_points        

        if clip_plane: # remove points behind the camera image plane
            cam_plane_z = max(self.K[0,0], self.K[1,1])
            z_filter = camera_points[2,:] > cam_plane_z
            camera_points = camera_points.T[z_filter].T

        proj_points = camera_points / camera_points[2]
        return proj_points.T[:,:2]

    def draw(self, ax: plt.Axes) -> None:
        self.basis.draw(ax)