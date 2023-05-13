import json
import os
import click
import numpy as np
import rasterio
from core.image_data import ImageDataList, ImageDataRecord
from shapely.geometry import Polygon, box
from PIL import Image
from utils import get_heights_tiff, get_terrain_heights

from utils.camera import Camera
from utils.utils import ensure_folder_exists

Image.MAX_IMAGE_PIXELS = 287944704
TILE_SIZE = 50

# def expand2square(pil_img, size=512, background_color=(0,0,0)):
#     width, height = pil_img.size
#     if width == height == size:
#         return pil_img
#     else:
#         result = Image.new(pil_img.mode, (size, size), background_color)
#         result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
#         return result


def prepare_tile(x: int, y:int, cameras, images: list[ImageDataRecord], output_folder: str):
    tile_name = f'{x}_{y}'
    folder_path = os.path.join(output_folder, tile_name, 'images')
    ensure_folder_exists(folder_path)
    
    l, b = x, y
    r, t = x+TILE_SIZE, y+TILE_SIZE
    tile_polygon_wc = box(l, b, r, t)

    tile_coords_wo_height = [[l, t], [r, t], [r, b], [l, b]]
    tile_coords = get_terrain_heights(tile_coords_wo_height)

    for cam_id in cameras.keys():
        # Iterate images from one cam at the time. Find the best (most centered) image and export it.
        best_image = None
        best_bbox = (1000000,)
        for im_data in images:
            if im_data.cam.cam_id.lower() != cam_id: continue
            if not tile_polygon_wc.intersects(im_data.bbox): continue

            xy = im_data.wc_to_ic(tile_coords)
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
        im = Image.open(best_image.path).crop(cropbox)
        im.save(os.path.join(folder_path, f'test_{best_image.name}_{"_".join((format(x, ".2f") for x in best_bbox))}.jpg'))
        im = Image.open(best_image.path).crop(cropbox).resize((512,512))
        im.save(os.path.join(folder_path, f'{best_image.name}_{"_".join((format(x, ".2f") for x in best_bbox))}.jpg'))


@click.command()
def prepare_analysis():
    #### Parameters ####
    config = 'config/lindesnes.json'
    test_area = {
        'minx': 412950,
        'maxx': 413200,
        'miny': 6452000,
        'maxy': 6452350
    }
    output_folder = 'analysis/test'
    #####################

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
    
    # Find image_data for images intersecting with the whole test_area
    test_area_polygon = box(test_area['minx'], test_area['miny'], test_area['maxx'], test_area['maxy'])
    images:list[ImageDataRecord] = list(filter(lambda im: test_area_polygon.intersects(im.bbox), image_data))
    
    # prepare for each tile...
    for x in range(test_area['minx'], test_area['maxx'], TILE_SIZE):
        if test_area['maxx'] - x < TILE_SIZE: print(f'Test area is extended to the right by {test_area["maxx"]-x} meters')

        for y in range(test_area['miny'], test_area['maxy'], TILE_SIZE):
            if test_area['maxy'] - y < TILE_SIZE: print(f'Test area is extended upwards by {test_area["maxy"]-y} meters')

            prepare_tile(x, y, cameras, images, output_folder)




if __name__ == '__main__':
    # config = 'data/config/grimstad_new.json'
    # x = 476155.42
    # y = 6465926.15
    # d = 25
    # aoi = box(x-d, y-d, x+d, y+d)
    # forward(config, aoi)
    # x = 413111.13
    # y = 6452289.41
    # d = 25


    prepare_analysis()