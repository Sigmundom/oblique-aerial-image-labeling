from typing import List
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import numpy as np

colors = ['blue', 'green', 'red']

class Building():
    def __init__(self):
        self.surface_vertices: List[np.ndarray] = []
        self.surface_types: List[int] = [] # 1: roof, 2: wall
        self.bbox = None

    def add_surface(self, vertices: np.ndarray, surfaceType: int):
        self.surface_vertices.append(vertices)
        self.surface_types.append(surfaceType)

    def calculate_bbox(self):
        vertices = np.concatenate(self.surface_vertices)
        x = vertices[:,0].min()
        x_max = vertices[:,0].max()
        y = vertices[:,1].min()
        y_max = vertices[:,1].max()
        self.bbox = (x, y, x_max, y_max)

    def draw(self, show=False):
        _, ax = plt.subplots()
        for surface_vertices, surface_type in zip(self.surface_vertices, self.surface_types):
            if surface_type == 2:
                ax.add_patch(Polygon(surface_vertices, color=colors[1]))
        for surface_vertices, surface_type in zip(self.surface_vertices, self.surface_types):
            if surface_type == 1:
                ax.add_patch(Polygon(surface_vertices, color=colors[0]))
        for surface_vertices, surface_type in zip(self.surface_vertices, self.surface_types):
            if surface_type == 3:
                ax.add_patch(Polygon(surface_vertices, color=colors[2]))
            
        if show:
            v = np.concatenate(self.surface_vertices)
            x_min, y_min = np.array(v)[:,:2].min(axis=0)
            x_max, y_max = np.array(v)[:,:2].max(axis=0)
                
            plt.axis([x_min, x_max, y_min, y_max])
            plt.show()

    def __getitem__(self, index):
        return self.surface_types[index], self.surface_types[index]

    def __str__(self):
        return f'Building with {len(self.surface_vertices)} surfaces and bbox {self.bbox}'