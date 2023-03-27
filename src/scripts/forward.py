import json
import os
import click
import numpy as np
from core.image_data import ImageData, ImageDataRecord
from shapely.geometry import Polygon, box
from PIL import Image
from utils import get_laser_data
from geotiff import GeoTiff

from utils.camera import Camera




# @click.command()
# @click.option('-c', '--config', default='data/config/lindesnes.json')
def forward(config: str, aoi: Polygon):
    with open(config, encoding='utf8') as f:
        config = json.load(f)

    cameras = {camera_info['cam_id'].lower(): Camera(camera_info) for camera_info in config['cameras']}
    image_paths = [os.path.join(folder, item) for folder in config['images'] for item in os.listdir(folder)]
    
    image_data_paths = config['image_data']

    image_data_format = config['image_data_format']
    if image_data_format == 'sos':
        image_data = ImageData.from_sos(image_paths, image_data_paths, cameras)
    elif image_data_format == 'dbf':
        image_data = ImageData.from_shp(image_paths, image_data_paths, cameras)
    else:
        raise ValueError(f'"{image_data_format}" is not a valid format. Must be "sos" or "dbf"')
    
    images:list[ImageDataRecord] = list(filter(lambda im: aoi.intersects(im.bbox), image_data))

    laser_data = get_laser_data(aoi)
    # laser_data = GeoTiff('aoi.tiff', as_crs=25832)

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
        left = minx + im_width//2
        top = im_height//2 - maxy
        right = maxx + im_width//2
        bottom = im_height//2 - miny
        # dx = (512-(right-left)) // 2
        # dy = (512-(bottom-top)) // 2
        m = 10
        im = Image.open(im_data.path).crop((left-m, top-m, right+m, bottom+m))
        im.save(f'test/{im_data.name}.jpg')


if __name__ == '__main__':
    config = 'data/config/grimstad_new.json'
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    forward(config, aoi)