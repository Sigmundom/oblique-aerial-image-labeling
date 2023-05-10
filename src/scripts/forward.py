import json
import os
import numpy as np
from core.image_data import ImageDataList, ImageDataRecord
from shapely.geometry import Polygon, box
from PIL import Image
from utils import get_laser_data

from utils.camera import Camera
from utils.utils import ensure_folder_exists

Image.MAX_IMAGE_PIXELS = 287944704

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
    elif image_data_format == 'shp':
        image_data = ImageDataList.from_shp(image_paths, image_data_paths, cameras)
    else:
        raise ValueError(f'"{image_data_format}" is not a valid format. Must be "sos" or "shp"')
    
    images:list[ImageDataRecord] = list(filter(lambda im: aoi.intersects(im.bbox), image_data))

    laser_data = get_laser_data(aoi)

    for cam_id in cameras.keys():
        # Iterate images from one cam at the time. Find the best (most centered) image and export it.
        best_image = None
        best_bbox = (1000000,)
        for im_data in images:
            if im_data.cam.cam_id.lower() != cam_id: continue
            l, b, r, t = laser_data.bounds
            heights = laser_data.read(1)
            # print(im_data.kappa, im_data.omega, im_data.phi)
            coords = np.array([
                [l, t, heights[0, 0]],
                [r, t, heights[0, 511]],
                [r, b, heights[511, 511]],
                [l, b, heights[511, 0]],
            ])
            xy = im_data.wc_to_ic(coords)
            minx, miny = xy.min(axis=0)
            maxx, maxy = xy.max(axis=0)

            bbox = (minx, miny, maxx, maxy)

            if sum(abs(x) for x in bbox) < sum(abs(x) for x in best_bbox):
                best_bbox = bbox
                best_image = im_data

        if best_image is None:
            raise Exception(f'No image from camera {cam_id} matching the area')

        im_height = best_image.cam.height_px
        im_width = best_image.cam.width_px

        minx, miny, maxx, maxy = best_bbox 
        left = round(minx + im_width//2)
        top = round(im_height//2 - maxy)
        right = round(maxx + im_width//2)
        bottom = round(im_height//2 - miny)
        cropbox = (left, top, right, bottom)
        im = Image.open(best_image.path).crop(cropbox).resize((512,512))
        # im.save(f'/home/sigmundmestad/code/Oblique-building-detection/Input/{best_image.name}_{"_".join(bbox)}.jpg')
        im.save(f'test/lindesnes_nadir_test/{best_image.name}_{"_".join((format(x, ".2f") for x in best_bbox))}.jpg')


if __name__ == '__main__':
    # config = 'data/config/grimstad_new.json'
    # x = 476155.42
    # y = 6465926.15
    # d = 25
    # aoi = box(x-d, y-d, x+d, y+d)
    # forward(config, aoi)

    config = 'data/config/lindesnes_nadir.json'
    x = 413111.13
    y = 6452289.41
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    forward(config, aoi)