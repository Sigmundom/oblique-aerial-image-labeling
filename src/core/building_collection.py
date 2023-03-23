import json
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
from shapely.strtree import STRtree
import shapely.geometry as sg
from utils import SurfaceType, get_municipality_border
from core.building import Building

def plot_polygon(polygon: sg.Polygon, color='blue'):
    arr = np.array(polygon.exterior.coords)
    plt.plot(arr[:,0], arr[:,1], c=color)


class WCBuildingCollection():
    def __init__(self, cityjson_path, municipality):
        with open(cityjson_path, encoding='utf8') as f:
            cityjson = json.load(f)
        self.city_objects = cityjson['CityObjects']
        self.vertices = np.array(cityjson['vertices'])
        self.outline = self._find_outline(municipality)
        self._buildings = None
        self._STRtree = None

    def _find_outline(self, municipality):
        municipality_border = get_municipality_border(municipality)
        cityjson_convex_hull = sg.MultiPoint(self.vertices[:,:2]).convex_hull       
        res = municipality_border.intersection(cityjson_convex_hull)
        return res

    def _get_buildings_and_STRtree(self):
        if self._STRtree is None:
            buildings = []
            t = default_timer()
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
            print(f'Create buildings took {default_timer()-t} seconds')
            t = default_timer()
            self._STRtree = STRtree([b.bbox_wc for b in buildings])
            print(f'Create STRtree {default_timer()-t} seconds')
            self._buildings = np.array(buildings)
        return self._buildings, self._STRtree

    def get_buildings_in_area(self, area: sg.Polygon):
        buildings, STRtree = self._get_buildings_and_STRtree()
        return list(buildings.take(STRtree.query(area)))


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
