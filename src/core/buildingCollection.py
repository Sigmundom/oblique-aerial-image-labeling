import json
from timeit import default_timer
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from shapely.strtree import STRtree
import shapely.geometry as sg
from utils import SurfaceType
from src.core.building import Building

def plot_polygon(polygon: sg.Polygon, color='blue'):
    arr = np.array(polygon.exterior.coords)
    plt.plot(arr[:,0], arr[:,1], c=color)


class WCBuildingCollection():
    def __init__(self, cityjson_path, municipality_geojson_path):
        self.municipality_geojson_path = municipality_geojson_path
        with open(cityjson_path, encoding='utf8') as f:
            cityjson = json.load(f)
        self.city_objects = cityjson['CityObjects']
        self.vertices = np.array(cityjson['vertices'])
        self._outline = None
        self._buildings = None
        self._STRtree = None

    @property
    def outline(self):
        if self._outline is not None:
            return self._outline
        with open(self.municipality_geojson_path, encoding='utf8') as f:
            geojson = json.load(f)
        municipality_border = sg.Polygon(geojson['omrade']['coordinates'][0][0])
        
        # cityjson_convex_hull = sg.MultiPoint(self.vertices[:,:2]).convex_hull
        # self._outline = municipality_border.intersection(cityjson_convex_hull)
        self._outline = municipality_border
        return self._outline        

    @property
    def STRtree(self):
        if self._STRtree is not None:
            return self._STRtree
        buildings = []
        for building in self.city_objects.values():
            geometry = building["geometry"]
            if len(geometry) > 1:
                print("Length is:", len(geometry))
            if len(geometry) == 0:
                continue
            
            boundaries = geometry[0]['boundaries']
            surface_types = [SurfaceType.parse(surface['type']) for surface in geometry[0]['semantics']['surfaces']]
            surfaces_wc = [np.take(self.vertices, boundary[0], axis=0) for boundary in boundaries]
            
            building = Building(surface_types, surfaces_wc)
            buildings.append(building)

        self._STRtree = STRtree([b.wc_bbox for b in buildings])
        self._buildings = np.array(buildings)

    def get_buildings_in_image(self, image_polygon: sg.Polygon):
        return list(self.buildings.take(self.STRtree.query(image_polygon)))


if __name__ == '__main__':
    # with open('grimstad.json', encoding='utf8') as f:
    #         grimstad = json.load(f)
    # a = np.array(grimstad['omrade']['coordinates'][0][0])
    # print(a.shape)
    # b = sg.Polygon(a)
    # c = np.array(b.exterior.coords)
    # print(c.shape)
    # n = np.array(a)
    # print(n)
    # plt.plot(a[:,0], a[:,1], c='red')
    # plt.show()
    # print(len(a), len(a[0]), len(a[0][0]))
    cityjson_path = 'data/3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json'
    municipality_geojson_path = 'grimstad.json'
    bc = WCBuildingCollection(cityjson_path, municipality_geojson_path)
