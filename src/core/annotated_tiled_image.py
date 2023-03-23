
import json
from PIL import Image
import numpy as np
from shapely.strtree import STRtree
from utils import ensure_folder_exists
from .annotated_tile import AnnotatedTile
from .building import Building
from .tiled_image import TiledImage

Image.MAX_IMAGE_PIXELS = 120000000

threshold_building_size = 500
threshold_building_part_size = 1000

class AnnotatedTiledImage(TiledImage):
    def __init__(
        self,
        buildings: list[Building],
        *args,
        tile_id_start=0,
        annotation_id_start=0,
        **kwargs ,
        ):
        super().__init__(*args, **kwargs)
        self.tile_id_start = tile_id_start
        self.annotation_id_start = annotation_id_start
        self.common_tile_info = {
            "height": self.tile_size[0],
            "width": self.tile_size[1],
        }
        for building in buildings:
            building.transform_to_image_coordinates(self.wc_to_ic)

        search_tree = STRtree([b.bbox_ic for b in buildings])
        buildings = np.array(buildings)
        self.tiles = [AnnotatedTile(tile, list(buildings.take(search_tree.query(tile.bbox)))) for tile in self]

    def export_semantic_segmentation(self, annotation_format, label_walls):
        self.save_image_data()
        ensure_folder_exists(f'{self.output_folder}/images')

        if annotation_format == 'mask':
            ensure_folder_exists(f'{self.output_folder}/labels') 
            for tile in self:
                if len(tile.buildings) == 0: continue
                tile.export_tile_with_label(label_walls)

        elif annotation_format == 'coco':
            raise NotImplementedError('Only support masks')
            # ensure_folder_exists(f'{self.output_folder}/annotations')
            # annotations = []
            # tiles = []
            # tile_id = self.tile_id_start
            # for tile in self:
            #     if len(tile.buildings) == 0: continue
            #     tile.save()
            #     tile_info, annotation = tile.create_coco_semantic_segmentation(tile_id, label_walls)
            #     annotations.extend(annotation)
            #     tiles.append(tile_info)
            #     tile_id +=1
        
            # coco = dict(
            #     info={},
            #     licenses=[],
            #     images=tiles,
            #     annotations=annotations
            # )
            # if label_walls:
            #     coco['categories'] = [{
            #             "id": 1,
            #             "name": "Roof",
            #         },
            #         {
            #             "id": 2,
            #             "name": "Wall"
            #         },
            #         {
            #             "id": 3,
            #             "name": "Terrace"
            #         },
            #         {
            #             "id": 4,
            #             "name": "Terrace wall"
            #         },
            #         {
            #             "id": 5,
            #             "name": "Handrail"
            #         }
            #         ]
            # else:
            #     coco['categories'] = [{
            #         "id": 1,
            #         "name": "Building"
            #     }]

            # with open(f'{self.output_folder}/annotations/segmentation.json', 'w', encoding='utf-8') as f:
            #     json.dump(coco, f)

