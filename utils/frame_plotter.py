import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PConf:
    limits2D: tuple = (-0.1, 12)
    limits3D: tuple = (0, 12)
    locator2D: int = 8
    locator3D: int = 4
    axis_labels: tuple = ('x','y','z')
    axis_colors: tuple = ('r','g','b')
    figsize: tuple = (10, 10)

    # (plane, (elev, azim, roll))
    views = {'XY':   (90, -90, 0),
             'XZ':    (0, -90, 0),
             'YZ':    (0,   0, 0),
             '-XY': (-90,  90, 0),
             '-XZ':   (0,  90, 0),
             '-YZ':   (0, 180, 0)}


class Frame:   

    def set_locator(axes, locator):
        axes.xaxis.set_major_locator(plt.MaxNLocator(locator))
        axes.yaxis.set_major_locator(plt.MaxNLocator(locator))
        axes.zaxis.set_major_locator(plt.MaxNLocator(locator)) 

    def mark_axis(axes, plane=None):
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z') 

        if plane is None:
            return
        else:
            if plane in ('XY', '-XY'):
                axes.set_zticklabels([])
                axes.set_zlabel('')
            if plane in ('XZ', '-XZ'):
                axes.set_yticklabels([])
                axes.set_ylabel('')
            if plane in ('YZ', '-YZ'):
                axes.set_xticklabels([])
                axes.set_xlabel('')

    @classmethod
    def make_3d_frame(cls, figure, subplot_index):
        ax = figure.add_subplot(*subplot_index, projection='3d')

        Frame.set_locator(ax, PConf.locator3D)   
        Frame.mark_axis(ax)
        ax.set_box_aspect(None, zoom=0.9)
        
        return cls(ax)

    @classmethod
    def make_2d_frame(cls, figure, subplot_index, plane=None):
        if (plane is None) or (plane not in PConf.views):
            raise ValueError('2d plane is not defined')  

        ax = figure.add_subplot(*subplot_index, projection='3d')
        
        Frame.set_locator(ax, PConf.locator2D)
        Frame.mark_axis(ax, plane)

        # ax.set_aspect('equal')
        ax.set_box_aspect((1,1,1), zoom=1.23)
        ax.set_proj_type('ortho')
        angles = PConf.views[plane]
        ax.view_init(elev=angles[0], azim=angles[1], roll=angles[2])

        return cls(ax)

    def __init__(self, axes):
        self.ax = axes        

        self.ax.set_xlim(*PConf.limits2D)
        self.ax.set_ylim(*PConf.limits2D)
        self.ax.set_zlim(*PConf.limits2D)


class View:

    def __init__(self):
        self.figure = plt.figure(figsize=PConf.figsize)
        self.figure.subplots_adjust(wspace=0, hspace=0)
        
        self.frames = [Frame.make_2d_frame(self.figure, (2, 2, 1), plane='XZ'), 
                       Frame.make_2d_frame(self.figure, (2, 2, 2), plane='YZ'),
                       Frame.make_2d_frame(self.figure, (2, 2, 3), plane='XY'),
                       Frame.make_3d_frame(self.figure, (2, 2, 4))]        

    def render(self, thing):
        for frame in self.frames:
            thing.draw(frame.ax)


class Scene:
    def __init__(self):
        self.view = View()
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def render(self):
        for element in self.elements:
            self.view.render(element)

        