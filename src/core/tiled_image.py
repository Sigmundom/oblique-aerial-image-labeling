from math import ceil
from os import path
from typing import List
import shapely.geometry as sg
from PIL import Image
from core.image_data import ImageDataRecord
from utils import get_image_data, ensure_folder_exists
from .tile import Tile

class TiledImage:
    def __init__(
            self, 
            image_data: ImageDataRecord,
            tile_size: int=512, 
            minimum_tile_overlap=0,
            output_folder='outputs'
            ):
        self.image_data = image_data
        self.wc_to_ic = image_data.wc_to_ic
        self.image_name = image_data.name
        if isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)
        else:
            self.tile_size = tile_size
        self.tile_overlap = minimum_tile_overlap
        self.output_folder = output_folder
        self.tiles = self._get_tiles()

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        return self.tiles[index]
    
        

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
    #     plt.savefig(f'{self.output_folder}/image_data/{self.image_name}.jpg')
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
                    crop_box = (j, i-tile_h, j+tile_w, i) # region in pixel coordinates for PIL.crop
                    tile_image = image.crop(crop_box)
                    bbox = sg.box(x, y, x+tile_w, y+tile_h) # Bbox in image coordinates
                    tiles.append(Tile(self, tile_image, crop_box, bbox))

        return tiles




if __name__ == '__main__':
    image_path = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    seamline_path = 'images/Somlinjefiler/cam4B.sos'
    image = Image.open(image_path)
    image_name = path.basename(image_path).split('.')[0]
    image_data = get_image_data(seamline_path, image_name)
    tiled_image = TiledImage(image, image_name, image_data)
    tiled_image.export_image_tiles()


