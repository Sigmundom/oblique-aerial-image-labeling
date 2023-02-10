from matplotlib.collections import PatchCollection
import shapely.geometry as sg
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt


import numpy as np

from utils import SurfaceType

colors = ['blue', 'green', 'red', 'yellow']

def remove_indices(original_list: list, indices:list):
    return [ele for idx, ele in enumerate(original_list) if idx not in indices]


class Building():
    def __init__(self, surface_types: list[SurfaceType], surfaces_wc: list[np.ndarray], surfaces_ic: list[np.ndarray]):
        self.surface_types = surface_types
        self.surfaces_wc = surfaces_wc
        self.surfaces_ic = surfaces_ic
        self.detect_terraces()

        # Create polygons and sort them based on surface type
        self.surfaces = {surface_type: [] for surface_type in SurfaceType}
        for surface_type, surface in zip(surface_types, surfaces_ic):
            self.surfaces[surface_type].append(sg.Polygon(surface))

        # Calculate bbox
        vertices = np.concatenate(self.surfaces_ic)
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)
        self.bbox = sg.box(x_min, y_min, x_max, y_max)

    def __getitem__(self, surface_type: SurfaceType):
        return self.surfaces[surface_type]


    def detect_terraces(self):
        auto_generated_handrails = [] # 75cm high wall surfaces
        handrail_lower_vertices = [] # To match against terrace surfaces 
        potential_terraces = [] # Flat roof surfaces
        confirmed_terraces = [] # Flat "roof surfaces" intersecting with a handrail surface and does not have a proper wall beneath.
        normal_walls = []
        potential_terrace_wall = []
      
        for i, (vertices, surface_type) in enumerate(zip(self.surfaces_wc, self.surface_types)):            
            if surface_type == SurfaceType.ROOF:
                if np.allclose(vertices[:,2], vertices[0,2]): # All vertices have same height
                    potential_terraces.append(i)
            elif surface_type == SurfaceType.WALL:
                surface_height = np.ptp(vertices[:,2])
                if np.isclose(surface_height, 0.75): # Auto-generated handrails are 75 cm tall
                    auto_generated_handrails.append(i)
                    # Saves the two lowest handrail vertices
                    handrail_lower_vertices.append(vertices[np.argpartition(vertices, 2)[:2]]) 
                else:
                    normal_walls.append(vertices)
                    if surface_height < 2.5:
                        potential_terrace_wall.append(i)

        if len(auto_generated_handrails) == 0: return

        for i in potential_terraces:
            # Not a terrace if it doesn't intersect with any auto-generated handrails
            if not any([np.any(np.all(np.equal(handrail_lower_vertices, vertex), axis=3)) for vertex in self.surfaces_wc[i]]):
                continue

            for vertex in self.surfaces_wc[i]:
                # lowest height value for all walls containing a vertex with common xy-coordinates as current potential terrace vertex
                intersecting_walls_height = [np.min(w[:, 2]) for w in normal_walls if np.any(np.all(np.equal(w[:,:2], vertex[:2]), axis=1))]
                if not any([vertex[2] - h > 1 for h in intersecting_walls_height]):
                    # If any of the potential terrace vertices does not have a proper wall beneath it is confirmed as a terrace.
                    confirmed_terraces.append(i)
                    break
                
        
        for i in auto_generated_handrails:
            self.surface_types[i] = SurfaceType.AUTO_GENERATED_HANDRAIL
            
        for i in confirmed_terraces:
            self.surface_types[i] = SurfaceType.TERRACE

        # Detect vertical surfaces only connected to a terrace and not a proper roof.
        roofs = [roof for roof, surface_type in zip(self.surfaces_wc, self.surface_types) if surface_type == SurfaceType.ROOF]
        for i in potential_terrace_wall:
            wall = self.surfaces_wc[i]
            if not any([np.any(np.all(np.equal(wall[0,:2], roof[:,:2]), axis=1)) for roof in roofs]):
                self.surface_types[i] = SurfaceType.TERRACE_WALL

                    


    def draw(self, show=False):
        ax = plt.gca()
        patches = []
        for surface in self[SurfaceType.TERRACE]:
            patches.append(Polygon(np.array(surface.exterior.coords), fc=colors[1], ec='black', alpha=0.5))
        for surface in self[SurfaceType.AUTO_GENERATED_HANDRAIL]:
            patches.append(Polygon(np.array(surface.exterior.coords), fc=colors[0], ec='black', alpha=0.5))
        for surface in self[SurfaceType.ROOF]:
            patches.append(Polygon(np.array(surface.exterior.coords), fc=colors[2], ec='black', alpha=0.5))
        for surface in self[SurfaceType.WALL]:
            patches.append(Polygon(np.array(surface.exterior.coords), fc=colors[3], ec='black', alpha=0.5))
            
        ax.add_collection(PatchCollection(patches))
        if show:
            v = np.concatenate(self.surfaces_ic)
            x_min, y_min = np.array(v)[:,:2].min(axis=0)
            x_max, y_max = np.array(v)[:,:2].max(axis=0)
                
            plt.axis([x_min, x_max, y_min, y_max])
            plt.show()

    def __str__(self):
        return f'Building with {len(self.surfaces_ic)} surfaces and bbox {self.bbox}'