
import json
from typing import List
from os import path
from PIL import Image
# from matplotlib import pyplot as plt
from rasterio.features import rasterize
from pycocotools.mask import encode, area, toBbox
import numpy as np
import shapely.geometry as sg
from building import Building
from create_annotation import create_annotation
from tiled_image import TiledImage, ensure_folder_exists
from utils.image_data import get_image_data

class AnnotatedTiledImage(TiledImage):
    def __init__(
        self,
        cityjson,
        *args,
        image_id_start=0,
        annotation_id_start=0,
        **kwargs ,
        ):
        super().__init__(*args, **kwargs)
        self.buildings = self._extract_buildings_from_cityjson(cityjson)
        self.image_id_start = image_id_start
        self.annotation_id_start = annotation_id_start

    def get_buildings_in_tile(self, tile_index) -> List[Building]:
        ax, ay = self.anchors[tile_index]
        a_height, a_width = self.tile_size
        def is_building_in_tile(building):
            bx_min, by_min, bx_max, by_max = building.bbox
            return not (
                bx_max < ax or
                by_max < ay - a_height or 
                bx_min > ax + a_width or 
                by_min > ay
                )

        return list(filter(is_building_in_tile, self.buildings))

    def export_semantic_segmentation(self):
        coco = self._create_annotation_base()
        coco['categories'] = [
            {
                "id": 1,
                "name": "Roof",
            },
            {
                "id": 2,
                "name": "Wall"
            }
        ]
        annotations = []
        annotation_id = self.annotation_id_start
        ensure_folder_exists('masks')
        for tile_index in range(len(self)):
            image_id = self.image_id_start + tile_index
            buildings = self.get_buildings_in_tile(tile_index)
            if len(buildings) == 0:
                continue
            roofs = []
            walls = []
            for building in buildings:
                surfaces = [self.ic_to_tc(vertices_ic, tile_index) for vertices_ic in building.surface_vertices]
                for surface, surface_type in zip(surfaces, building.surface_types):
                    if surface_type == 1:
                        roofs.append(sg.Polygon(surface))
                    else: 
                        walls.append(sg.Polygon(surface))
            mask = np.zeros(self.tile_size, dtype=np.uint8)
            rasterize(walls, default_value=2, out_shape=self.tile_size, out=mask)
            rasterize(roofs, default_value=1, out_shape=self.tile_size, out=mask)
            wall_rle = encode(np.asfortranarray(mask == 2))
            wall_rle['counts'] = wall_rle['counts'].decode('utf-8')
            roof_rle = encode(np.asfortranarray(mask == 1))
            roof_rle['counts'] = roof_rle['counts'].decode('utf-8')
            annotations.append({
                'image_id': image_id,
                'category_id': 1,
                'segmentation': roof_rle,
                'id': annotation_id,
                'area': int(area(roof_rle)),
                'bbox': toBbox(roof_rle).tolist()
            })
            annotation_id += 1
            annotations.append({
                'image_id': image_id,
                'category_id': 2,
                'segmentation': wall_rle,
                'id': annotation_id,
                'area': int(area(wall_rle)),
                'bbox': toBbox(wall_rle).tolist()
            })
            annotation_id += 1

        coco['annotations'] = annotations

        ensure_folder_exists(f'{self.output_folder}/annotations')
        with open(f'{self.output_folder}/annotations/segmentation.json', 'w', encoding='utf-8') as f:
            json.dump(coco, f)


    def export_instance_segmentation(self):
        coco = self._create_annotation_base()
        coco['categories'] = [
            {
                "id": 1,
                "name": "Builing",
            }
        ]
        annotations = []
        annotation_id = self.annotation_id_start
        for tile_index in range(len(self)):
            image_id = self.image_id_start + tile_index
            buildings = self.get_buildings_in_tile(tile_index)
            if len(buildings) == 0:
                continue
            for building in buildings:
                surfaces = [self.ic_to_tc(vertices_ic, tile_index) for vertices_ic in building.surface_vertices]
                annotation = create_annotation(surfaces, image_id, annotation_id)
                if annotation is not None:
                    annotations.append(annotation)
                annotation_id += 1

        coco['annotations'] = annotations

        ensure_folder_exists(f'{self.output_folder}/annotations')
        with open(f'{self.output_folder}/annotations/instances_train.json', 'w', encoding='utf8') as f:
            json.dump(coco, f)

    def _create_annotation_base(self):
        return dict(
            info={},
            licenses=[],
            images=[
                {
                    "id": self.image_id_start + i,
                    "height": self.tile_size[0],
                    "width": self.tile_size[1],
                    "file_name": f'{self.image_name}_{i}.jpg',
                    "date_captured": str(self.get_date_captured())
                } for i in range(len(self))
            ],
        )

    def _extract_buildings_from_cityjson(self, cityjson) -> List[Building]:
        corner_points = ["UL", "UR", "LR", "LL"]
        x_coords = [float(self.image_data[f'{point}x']) for point in corner_points]
        y_coords = [float(self.image_data[f'{point}y']) for point in corner_points]
        x, x_max = min(x_coords), max(x_coords)
        y, y_max = min(y_coords), max(y_coords)

        all_buildings = cityjson['CityObjects']
        all_vertices = np.array(cityjson['vertices'])
        vertices_in_image = (all_vertices[:, 0] > x) & (all_vertices[:,0] < x_max) & (all_vertices[:,1] > y) & (all_vertices[:, 1] < y_max)
        buildings_in_image = []

        for building in all_buildings.values():
            geometry = building["geometry"]
            if len(geometry) > 1:
                print("Length is:", len(geometry))
            if len(geometry) == 0:
                continue
            boundaries = geometry[0]['boundaries']
            vertices_i = [v for boundary in boundaries for v in boundary[0]]
            
            if np.any(np.take(vertices_in_image, vertices_i, 0)):
                building = Building()
                surfaces = geometry[0]['semantics']['surfaces']
                for boundary, surface in zip(boundaries, surfaces):
                    surface_type = 1 if surface['type'] == 'RoofSurface' else 2 # Assuming there are only 'RoofSurface' and 'WallSurface'
                    vertices = np.take(all_vertices, boundary[0], axis=0)
                    vertices_ic = self.wc_to_ic(vertices)
                    building.add_surface(vertices_ic, surface_type)
                building.calculate_bbox()
                buildings_in_image.append(building)
        
        return buildings_in_image


if __name__ == '__main__':
    image_path = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    seamline_path = 'images/Somlinjefiler/cam4B.sos'
    image = Image.open(image_path)
    image_name = path.basename(image_path).split('.')[0]
    image_data = get_image_data(seamline_path, image_name)
    print('Loading CityGML', end='')
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        cityjson = json.load(f)
    print(' - Complete')

    tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder='outputs/test')
    # tiled_image.export_image_tiles()
    tiled_image.export_semantic_segmentation()
    # tiled_image.export_instance_segmentation()