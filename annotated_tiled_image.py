
from timeit import default_timer
from os import path
import json
from PIL import Image
import click
import numpy as np
import shapely.geometry as sg
from shapely.strtree import STRtree
from annotated_tile import AnnotatedTile
from building import Building
from enums import SurfaceType
from tiled_image import TiledImage, ensure_folder_exists
from utils import get_image_data

Image.MAX_IMAGE_PIXELS = 120000000

threshold_building_size = 500
threshold_building_part_size = 1000

class AnnotatedTiledImage(TiledImage):
    def __init__(
        self,
        cityjson,
        *args,
        tile_id_start=0,
        annotation_id_start=0,
        **kwargs ,
        ):
        super().__init__(*args, **kwargs)
        self._annotate_tiles(cityjson)
        self.tile_id_start = tile_id_start
        self.annotation_id_start = annotation_id_start
        self.common_tile_info = {
            "height": self.tile_size[0],
            "width": self.tile_size[1],
            "date_captured": str(self.get_date_captured())
        }

    def _annotate_tiles(self, cityjson):
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
                surface_types = [SurfaceType.parse(surface['type']) for surface in geometry[0]['semantics']['surfaces']]
                surfaces_wc = [np.take(all_vertices, boundary[0], axis=0) for boundary in boundaries]
                surfaces_ic = [self.wc_to_ic(s) for s in surfaces_wc]
                building = Building(surface_types, surfaces_wc, surfaces_ic)
                buildings_in_image.append(building)
        
        building_STRtree = STRtree([b.bbox for b in buildings_in_image])
        buildings_in_image = np.array(buildings_in_image)
        
        self.tiles = [AnnotatedTile(tile, list(buildings_in_image.take(building_STRtree.query(tile.bbox)))) for tile in self]


    def export_semantic_segmentation(self, annotation_format, label_walls):
        self.save_image_data()
        ensure_folder_exists(f'{self.output_folder}/images')

        if annotation_format == 'mask':
            ensure_folder_exists(f'{self.output_folder}/labels') 
            for tile in self:
                if len(tile.buildings) == 0: continue
                tile.export_tile_with_label(label_walls)

        elif annotation_format == 'coco':
            ensure_folder_exists(f'{self.output_folder}/annotations')
            annotations = []
            tiles = []
            tile_id = self.tile_id_start
            for tile in self:
                if len(tile.buildings) == 0: continue
                tile.save()
                tile_info, annotation = tile.create_coco_semantic_segmentation(tile_id, label_walls)
                annotations.extend(annotation)
                tiles.append(tile_info)
                tile_id +=1
        
            coco = dict(
                info={},
                licenses=[],
                images=tiles,
                annotations=annotations
            )
            if label_walls:
                coco['categories'] = [{
                        "id": 1,
                        "name": "Roof",
                    },
                    {
                        "id": 2,
                        "name": "Wall"
                    },
                    {
                        "id": 3,
                        "name": "Terrace"
                    },
                    {
                        "id": 4,
                        "name": "Terrace wall"
                    },
                    {
                        "id": 5,
                        "name": "Handrail"
                    }
                    ]
            else:
                coco['categories'] = [{
                    "id": 1,
                    "name": "Building"
                }]

            with open(f'{self.output_folder}/annotations/segmentation.json', 'w', encoding='utf-8') as f:
                json.dump(coco, f)

@click.command()
@click.argument('image_path')
@click.argument('seamline_path')
@click.option('-o', '--output-folder', default='outputs')
@click.option('-ts', '--tile-size', default=512) 
@click.option('-f','--annotation-format', default='mask', type=click.Choice(['mask', 'coco'], case_sensitive=False))
@click.option('-w', '--label-walls', is_flag=True)
def semantic_segmentation(image_path, seamline_path, output_folder, tile_size, annotation_format, label_walls):
    t_start = default_timer()
    print('Loading CityGML', end='')
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        cityjson = json.load(f)
    print(' - Complete - Took', default_timer() - t_start, 'seconds' )

    t = default_timer()
    image_name = path.basename(image_path).split('.')[0]
    print('Processing ', image_name, '...')
    image = Image.open(image_path)
    image_data = get_image_data(seamline_path, image_name)
    
    tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(tile_size, tile_size))

    print('Exporting semantic segmentation', end=' - ')
    t = default_timer()
    tiled_image.export_semantic_segmentation(annotation_format, label_walls)
    print(default_timer() -t, 'seconds')
