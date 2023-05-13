import json
from math import ceil
from os import path
from typing import List
import numpy as np
import requests
import shapely.geometry as sg
from PIL import Image
from core.image_data import ImageDataRecord
from utils import get_image_data, ensure_folder_exists
from core.tile import Tile
from utils.get_terrain_heights import get_terrain_heights




class TiledImage:
    def __init__(
            self, 
            image_data: ImageDataRecord,
            tile_size: int=512, 
            minimum_tile_overlap=0,
            output_folder='outputs',
            exclude_area=None,
            include_area=None
            ):
        self.image_data = image_data
        self.wc_to_ic = image_data.wc_to_ic

        self.exclude_area_polygon = self._get_area_polygon(exclude_area)
        self.include_area_polygon = self._get_area_polygon(include_area)

        if isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)
        else:
            self.tile_size = tile_size
        self.tile_overlap = minimum_tile_overlap
        self.output_folder = output_folder
        if include_area is not None and self.include_area_polygon is None:
            self.tiles = []
        else:
            self.tiles = self._get_tiles()

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        return self.tiles[index]
    
    def _get_area_polygon(self, areas):
        if areas is None: return None
        
        def get_polygon(area):
            if area['type'] != 'Polygon':
                raise NotImplementedError('Only supports geojson polygon')
            if len(area['coordinates']) > 1:
                raise NotImplementedError('only supports simple Polygons')
            
            coordinates_wc = np.array(area['coordinates'][0])
            assert coordinates_wc.shape[1] == 3
            
            polygon = sg.Polygon(self.wc_to_ic(coordinates_wc))
            if not polygon.is_valid:
                polygon = polygon.buffer(0.0)
                if not polygon.is_valid:
                    print('INVALID POLYGON:', polygon)
                    return sg.Polygon()
            return polygon
                
        if isinstance(areas, list):
            area_polygon = sg.MultiPolygon((get_polygon(area) for area in areas))
        else:
            area_polygon = get_polygon(areas)

        dx = self.image_data.cam.width_px // 2
        dy = self.image_data.cam.height_px // 2
        image_polygon = sg.box(-dx, -dy, dx, dy)

        if not area_polygon.intersects(image_polygon):
            return None
        
        return area_polygon.intersection(image_polygon)

    # def save_tile_map(self) -> None:
    #     ax = plt.gca()
    #     im_h, im_w = self.image.height, self.image.width
    #     ax.imshow(self.image, extent=[-im_w/2., im_w/2., -im_h/2., im_h/2. ])
    #     for i, anchor in enumerate(self.anchors):
    #         x, y = anchor
    #         h = self.tile_size[0]
    #         y = y-h
    #         ax.add_patch(Rectangle((x,y), width=self.tile_size[1], height=self.tile_size[0], linewidth=1, edgecolor='r', facecolor='none'))
    #         ax.text(x,y, str(i))
    #     plt.savefig(f'{self.output_folder}/image_data/{self.image_data.name}.jpg')
    #     plt.close()

    def save_image_data(self):
        folder = f'{self.output_folder}/image_data'
        ensure_folder_exists(folder)
        self.image_data.save_image_data(folder, self.tile_size)

    def export_image_tiles(self) -> None:
        # Save image data
        self.save_image_data()

        # Save tile jpgs
        ensure_folder_exists(f'{self.output_folder}/images')
        for tile in self:
            tile.save()

        # self.save_tile_map()


    # def get_date_captured(self) -> datetime:
    #     date, time = self.image_data['ShotDate']
    #     year, month, day = [int(x) for x in date.split('/')]
    #     h = int(time[:2])
    #     m = int(time[2:4])
    #     s = int(time[4:6])
    #     return datetime(year, month, day, h, m, s)

    def _get_tiles(self) -> List[Tile]:
        with Image.open(self.image_data.path) as image: 
            im_h, im_w = image.height, image.width
            tile_h, tile_w = self.tile_size
            num_tiles = (ceil(im_h / (tile_h-self.tile_overlap)), ceil(im_w / (tile_w-self.tile_overlap)))
        
            step = (ceil((im_h-tile_h) / num_tiles[0]), ceil((im_w-tile_w) / num_tiles[1]))

            tiles = []

            for tile_row in range(num_tiles[0]+1):
                i = im_h-tile_row*step[0] if tile_row < num_tiles[0] else tile_h
                y = -i + im_h//2 
                for tile_col in range(num_tiles[1]+1):
                    j = tile_col*step[1] if tile_col < num_tiles[1] else im_w-tile_w
                    x = j-im_w//2
                    bbox = sg.box(x, y, x+tile_w, y+tile_h) # Bbox in image coordinates

                    if bbox.intersects(self.exclude_area_polygon): continue
                    if self.include_area_polygon is not None and not bbox.intersection(self.include_area_polygon).area / bbox.area > 0.75: continue

                    crop_box = (j, i-tile_h, j+tile_w, i) # region in pixel coordinates for PIL.crop
                    tile_image = image.crop(crop_box)
                    tiles.append(Tile(self, tile_image, crop_box, bbox))
        return tiles




if __name__ == '__main__':
    coordinates = [
        [
            412940.28756845824,
            6452011.80591097
        ],
        [
            412947.64909195655,
            6452346.768161283
        ],
        [
            413210.5612016448,
            6452340.998732395
        ],
        [
            413203.22191662807,
            6452006.036181933
        ],
        [
            412940.28756845824,
            6452011.80591097
        ]
    ]
    heights = get_terrain_heights(coordinates)
    print(heights)


