import math
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from utils import SurfaceType, create_coco_rle_annotation
from .building import Building 
from .tile import Tile

class AnnotatedTile(Tile):  
    def __init__(self, tile:Tile, buildings:list[Building]):
        super().__init__(tile.parent, tile.crop_box, tile.bbox)
        self.buildings = buildings

    def get_all_surfaces_of_type(self, surface_type:SurfaceType):
        surfaces = []
        for building in self.buildings:
            surfaces.extend([self.ic_to_tc(surface) for surface in building[surface_type]])
        return surfaces

    def _create_mask(self, label_walls:bool):
        tile_size = self.parent.tile_size
        mask = np.zeros(tile_size, dtype=np.uint8)
        walls = self.get_all_surfaces_of_type(SurfaceType.WALL)
        roofs = self.get_all_surfaces_of_type(SurfaceType.ROOF)

        if label_walls:
            if len(walls) > 0:
                rasterize(walls, default_value=2, out_shape=tile_size, out=mask)
            if len(roofs) > 0:
                rasterize(roofs, default_value=1, out_shape=tile_size, out=mask)
        else:
            if len(walls) > 0:
                rasterize(self.get_all_surfaces_of_type(SurfaceType.WALL), default_value=1, out_shape=tile_size, out=mask)
            if len(roofs) > 0:
                rasterize(self.get_all_surfaces_of_type(SurfaceType.ROOF), default_value=1, out_shape=tile_size, out=mask)
        return mask

    def create_coco_semantic_segmentation(self, tile_id:int, label_walls:bool):
        mask = self._create_mask(label_walls)
        if label_walls:
            wall_mask = mask == 2
            roof_mask = mask == 1
        
            annotations = []
            if (np.any(roof_mask)):
                annotations.append(create_coco_rle_annotation(tile_id, 1, roof_mask))
            if (np.any(wall_mask)):
                annotations.append(create_coco_rle_annotation(tile_id, 2, wall_mask))
        else:
            annotations = [create_coco_rle_annotation(tile_id, 1, mask)]

        tile_info = {
            **self.parent.common_tile_info,
            "id": tile_id,
            "file_name": f'{repr(self)}.jpg',
        }

        return tile_info, annotations


    def export_tile_with_label(self, label_walls):
        """Exports the tile as jpg in '/images' and label as png in /labels"""
        mask = self._create_mask(label_walls)
        if mask.sum() / math.prod(self.parent.tile_size) > .05:
            self.save()
            plt.imsave(f'{self.parent.output_folder}/labels/{repr(self)}.png', mask, cmap=cm.gray)
