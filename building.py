from typing import List

import numpy as np


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

    def __getitem__(self, index):
        return self.surface_types[index], self.surface_types[index]

    def __str__(self):
        return f'Building with {len(self.surface_vertices)} surfaces and bbox {self.bbox}'