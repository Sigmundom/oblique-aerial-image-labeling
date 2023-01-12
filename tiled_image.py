from datetime import datetime
import json
from math import ceil
from os import path, makedirs
from typing import List
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from utils import get_image_data
from utils.transformers.wc_to_ic import get_wc_to_ic_transformer
from utils.types import Anchor

def ensure_folder_exists(folder):
    if not path.isdir(folder):
        makedirs(folder)

class TiledImage:
    def __init__(
            self, 
            image,
            image_name, 
            image_data, 
            tile_size=(1024,1024), 
            minimum_tile_overlap=0,
            output_folder='outputs'
            ):
        self.image = image
        self.image_name = image_name
        self.image_data = image_data
        self.tile_size = tile_size
        self.tile_overlap = minimum_tile_overlap
        self.output_folder = output_folder
        self.wc_to_ic = get_wc_to_ic_transformer(image_data)
        self.anchors = self._get_tile_anchors()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        return self.get_tile(self.anchors[index])

    def ic_to_tc(self, image_xy: np.ndarray, tile_index:int) -> np.ndarray:
        assert image_xy.ndim == 2
        ax, ay = self.anchors[tile_index]
        tile_xy = np.empty_like(image_xy)
        tile_xy[:,0] = image_xy[:,0] - ax
        tile_xy[:,1] = ay - image_xy[:,1]
        return tile_xy


    def export_image_tiles(self) -> None:
        ensure_folder_exists(f'{self.output_folder}/images')
        ensure_folder_exists(f'{self.output_folder}/image_data')
        for i, tile in enumerate(self):
            tile_name = f'{self.image_name}_{i}'
            tile.save(f'{self.output_folder}/images/{tile_name}.jpg')
        
        self.image_data['tiles'] = dict(size=self.tile_size, anchors=self.anchors)
        with open(f'{self.output_folder}/image_data/{self.image_name}.json', 'w', encoding='utf8') as f:
            json.dump(self.image_data, f)

        ax = plt.gca()
        im_h, im_w = self.image.height, self.image.width
        ax.imshow(self.image, extent=[-im_w/2., im_w/2., -im_h/2., im_h/2. ])
        for i, anchor in enumerate(self.anchors):
            x, y = anchor
            h = self.tile_size[0]
            y = y-h
            ax.add_patch(Rectangle((x,y), width=self.tile_size[1], height=self.tile_size[0], linewidth=1, edgecolor='r', facecolor='none'))
            ax.text(x,y, str(i))

        plt.savefig(f'{self.output_folder}/image_data/{self.image_name}.png')
        plt.close()

    def get_date_captured(self) -> datetime:
        date, time = self.image_data['ShotDate']
        year, month, day = [int(x) for x in date.split('/')]
        h = int(time[:2])
        m = int(time[2:4])
        s = int(time[4:6])
        return datetime(year, month, day, h, m, s)

    def get_tile(self, anchor: Anchor) -> Image:
        im_w, im_h = self.image.size
        h, w = self.tile_size
        x, y = anchor
        i = im_h//2 - y
        j = x + im_w//2
        box = (j, i, j+w, i+h)
        return self.image.crop(box)
    
    def _get_tile_anchors(self) -> List[Anchor]:
        im_h, im_w = self.image.height, self.image.width
        tile_h, tile_w = self.tile_size
        num_tiles = (ceil(im_h / (tile_h-self.tile_overlap)), ceil(im_w / (tile_w-self.tile_overlap)))
        
        step = (ceil((im_h-tile_h) / num_tiles[0]), ceil((im_w-tile_w) / num_tiles[1]))

        anchor_points = []

        for tile_row in range(num_tiles[0]+1):
            y = -tile_row*step[0] if tile_row < num_tiles[0] else -im_h+tile_h
            y += im_h//2
            for tile_col in range(num_tiles[1]+1):
                x = tile_col*step[1] if tile_col < num_tiles[1] else im_w-tile_w
                x -= im_w//2
                anchor_points.append((x,y))

        return anchor_points




if __name__ == '__main__':
    image_path = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    seamline_path = 'images/Somlinjefiler/cam4B.sos'
    image = Image.open(image_path)
    image_name = path.basename(image_path).split('.')[0]
    image_data = get_image_data(seamline_path, image_name)
    tiled_image = TiledImage(image, image_name, image_data)
    tiled_image.export_image_tiles()


