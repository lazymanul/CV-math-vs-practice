import numpy as np
import matplotlib.pyplot as plt

class Basis:

    def __init__(self,                  
                 name='W',                 
                 origin=np.zeros(3),
                 basis=np.eye(3),
                 parent_basis=None):
        self.name = name
        self.origin = origin
        self.basis = basis
        self.parent_basis = parent_basis
        self.child_bases = [] 

    def draw(self, ax):
        ax.quiver(*self.origin, *self.basis[:,0], color='r')
        ax.quiver(*self.origin, *self.basis[:,1], color='g')
        ax.quiver(*self.origin, *self.basis[:,2], color='b')

    def get_transform(self):
        T = np.zeros((4,4))
        T[:3,:3] = self.basis
        T[:3,3] = self.origin
        T[3,3] = 1
        return T


class Point:

    def __init__(self,                  
                 name='P',
                 coords=np.ones(3),
                 basis=None):
        self.name = name
        self.coords = coords
        self.basis = basis

    def draw(self, ax):
        if self.basis is None:
            raise AttributeError('point doesn\'t have a basis')
        else:            
            T = self.basis.get_transform()            
            hom_coords = np.hstack((self.coords, [1]))
            world_coords = (T @ hom_coords)[:3]            
            ax.plot(*world_coords, marker='o', color='black', markersize=5)