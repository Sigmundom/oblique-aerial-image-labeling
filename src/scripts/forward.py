import json
import os
import numpy as np
from core.image_data import ImageDataList, ImageDataRecord
from shapely.geometry import Polygon, box
from PIL import Image
from utils import get_laser_data

from utils.camera import Camera
from utils.utils import ensure_folder_exists

def expand2square(pil_img, size=512, background_color=(0,0,0)):
    width, height = pil_img.size
    if width == height == size:
        return pil_img
    else:
        result = Image.new(pil_img.mode, (size, size), background_color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result



# @click.command()
# @click.option('-c', '--config', default='data/config/lindesnes.json')
def forward(config: str, aoi: Polygon):
    ensure_folder_exists('test')
    with open(config, encoding='utf8') as f:
        config = json.load(f)

    cameras = {camera_info['cam_id'].lower(): Camera(camera_info) for camera_info in config['cameras']}
    image_paths = [os.path.join(folder, item) for folder in config['images'] for item in os.listdir(folder)]
    
    image_data_paths = config['image_data']

    image_data_format = config['image_data_format']
    if image_data_format == 'sos':
        image_data = ImageDataList.from_sos(image_paths, image_data_paths, cameras)
    elif image_data_format == 'dbf':
        image_data = ImageDataList.from_shp(image_paths, image_data_paths, cameras)
    else:
        raise ValueError(f'"{image_data_format}" is not a valid format. Must be "sos" or "dbf"')
    
    images:list[ImageDataRecord] = list(filter(lambda im: aoi.intersects(im.bbox), image_data))

    laser_data = get_laser_data(aoi)

    for im_data in images:
        im = Image.open(im_data.path)
        l, b, r, t = laser_data.bounds
        heights = laser_data.read(1)
        coords = np.array([
            [l, t, heights[0, 0]],
            [r, t, heights[0, 511]],
            [r, b, heights[511, 511]],
            [l, b, heights[511, 0]],
        ])
        xy = im_data.wc_to_ic(coords)
        minx, miny = xy.min(axis=0)
        maxx, maxy = xy.max(axis=0)
        im_height = im_data.cam.height_px
        im_width = im_data.cam.width_px
        left = round(minx + im_width//2)
        top = round(im_height//2 - maxy)
        right = round(maxx + im_width//2)
        bottom = round(im_height//2 - miny)
        crop_box = (left, top, right, bottom)
        im = Image.open(im_data.path).crop(crop_box).resize((512,512))
        # im = expand2square(im)
        bbox = [str(round(x, ndigits=4)) for x in [minx, maxy, maxx, miny]]
        im.save(f'/home/sigmundmestad/code/Oblique-building-detection/Input/{im_data.name}_{"_".join(bbox)}.jpg')


if __name__ == '__main__':
    config = 'data/config/grimstad_new.json'
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    forward(config, aoi)