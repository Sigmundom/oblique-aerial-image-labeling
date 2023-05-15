import json
import math
import os
import click
import numpy as np
from core.image_data import ImageDataList, ImageDataRecord
from shapely.geometry import Polygon, box
from PIL import Image
from utils import get_terrain_heights

from utils.camera import Camera
from utils.utils import ensure_folder_exists

Image.MAX_IMAGE_PIXELS = 287944704
TILE_SIZE_M = 50
TILE_SIZE_PX = 512

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
    folder_path = os.path.join(output_folder, tile_name)
    images_folder_path = os.path.join(folder_path, 'images')
    masks_folder_path = os.path.join(folder_path, 'masks')

    ensure_folder_exists(images_folder_path)
    ensure_folder_exists(masks_folder_path)

    
    l, b = x, y
    r, t = x+TILE_SIZE_M, y+TILE_SIZE_M
    tile_polygon_wc = box(l, b, r, t)

    tile_coords_wo_height = [[l, t], [r, t], [r, b], [l, b]]
    tile_coords = get_terrain_heights(tile_coords_wo_height)

    image_info = {}

    for cam_id in cameras.keys():
        # Iterate images from one cam at the time. Find the best (most centered) image and export it.
        best_image = None
        best_bbox = (1000000,)
        for im_data in images:
            if im_data.cam.cam_id != cam_id: continue
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

        minx, miny, maxx, maxy = [round(x) for x in best_bbox] 

        tile_width = maxx - minx
        tile_height = maxy - miny

        cropbox_size = max(TILE_SIZE_PX, tile_width, tile_height)

        left = minx + im_width//2
        right = maxx + im_width//2
        top = im_height//2 - maxy
        bottom = im_height//2 - miny


        dx = (cropbox_size - tile_width) / 2
        dy = (cropbox_size - tile_height) / 2

        left -= math.floor(dx)
        right += math.ceil(dx)
        top -= math.floor(dy)
        bottom += math.ceil(dy)
        
        assert right - left == bottom - top == cropbox_size

        cropbox = (left, top, right, bottom)
        im = Image.open(best_image.path).crop(cropbox)

        if cropbox_size != TILE_SIZE_PX:
            im = im.resize((TILE_SIZE_PX, TILE_SIZE_PX))
        im.save(os.path.join(images_folder_path, f'{best_image.name}.jpg'))

        image_info[cam_id] = {
            'image_name': best_image.name,
            'bbox_ic': best_bbox,
            'cropbox_size': cropbox_size,
            'dx': dx,
            'dy': dy
        }

    tile_info =  {
        'x': x,
        'y': y,
        'image_info': image_info 
    }
    with open(os.path.join(folder_path, 'info.json'), 'x') as f:
        json.dump(tile_info, f)
    
        # left = round(minx + im_width//2)
        # top = round(im_height//2 - maxy)
        # right = round(maxx + im_width//2)
        # bottom = round(im_height//2 - miny)
        # cropbox = (left, top, right, bottom)
        # im = Image.open(best_image.path).crop(cropbox)
        # im.save(os.path.join(folder_path, f'test_{best_image.name}_{"_".join((format(x, ".2f") for x in best_bbox))}.jpg'))
        # im = Image.open(best_image.path).crop(cropbox).resize((512,512))
        # im.save(os.path.join(folder_path, f'{best_image.name}_{"_".join((format(x, ".2f") for x in best_bbox))}.jpg'))


@click.command()
def prepare_analysis():
    #### Parameters ####
    config = 'config/analysis/lindesnes.json'
    # test_area = {
    #     'minx': 412950,
    #     'maxx': 413200,
    #     'miny': 6452000,
    #     'maxy': 6452350
    # }
    #####################

    with open(config, encoding='utf8') as f:
        config = json.load(f)

    output_folder = config['folder']
    cameras = {camera_info['cam_id']: Camera(camera_info) for camera_info in config['cameras']}
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
    test_area_coordinates = np.array(config['test_area']['coordinates'][0])
    test_area_polygon = Polygon(test_area_coordinates)

    # test_area_polygon = box(test_area['minx'], test_area['miny'], test_area['maxx'], test_area['maxy'])
    images:list[ImageDataRecord] = list(filter(lambda im: test_area_polygon.intersects(im.bbox), image_data))
    
    minx, miny = test_area_coordinates.min(axis=0)
    maxx, maxy = test_area_coordinates.max(axis=0)
    # prepare for each tile...
    for x in range(minx, maxx, TILE_SIZE_M):
        if maxx - x < TILE_SIZE_M: print(f'Test area is extended to the right by {maxx-x} meters')

        for y in range(miny, maxy, TILE_SIZE_M):
            if maxy - y < TILE_SIZE_M: print(f'Test area is extended upwards by {maxy-y} meters')

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