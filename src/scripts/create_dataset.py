import json
from typing import OrderedDict
from os import path
import os
from timeit import default_timer
import click
import numpy as np
from PIL import Image
import shapely.geometry as sg

from core import AnnotatedTiledImage, WCBuildingCollection
from libs.sosi import read_sos

def get_image_bbox(image_extent: OrderedDict):
    image_bbox = np.array(list(image_extent.values())[0], dtype=np.float64)
    image_bbox[:, [1, 0]] = image_bbox[:, [0, 1]]

    image_bbox /= 100 # Coordinates are given in cm for some reason. Convert to m.
    return sg.Polygon(image_bbox)


def is_image_in_area(image_data: dict, image_bbox: sg.Polygon, area: sg.Polygon, available_images: list[str]) -> bool:
    if not "ImageName" in image_data: return False
                
    image_name = image_data["ImageName"]
    if not f'{image_name}.jpg' in available_images: return False

    
    intersection = image_bbox.intersection(area)
    overlap = intersection.area / image_bbox.area
    if overlap > 0.99:
        return True
    elif overlap > 0.5:
        print('Partial overlap:', overlap)
        return False

def process_image(images_path, image_data, buildings_in_image, config):
    image_name = image_data['ImageName']

    print('Processing ', image_name, '...')
    image = Image.open(path.join(images_path, f'{image_name}.jpg'))
    tiled_image = AnnotatedTiledImage(buildings_in_image, image, image_name, image_data, output_folder=config['output_folder'], tile_size=config['tile_size'])

    print('Exporting semantic segmentation', end=' - ')
    t = default_timer()
    annotation_format = config['annotation_format']
    label_walls = config['label_walls']
    tiled_image.export_semantic_segmentation(annotation_format, label_walls)
    print(default_timer() -t, 'seconds')


@click.command()
@click.option('-c', '--config', default='data/config/grimstad.json')
def create_dataset(config):
    with open(config, encoding='utf8') as f:
        config = json.load(f)

    for area in config['areas']:
        buildings = WCBuildingCollection(area['cityjson'], area['municipality'])
        area_outline = buildings.outline

        for data_source in config['data']:
            seamline = data_source['seamline']
            images_path = data_source['images']
            available_images = os.listdir(images_path)
            seamline_file = list(read_sos(seamline).values())
            # images = []
            for i in range(1,len(seamline_file)-1,2):
                image_data = seamline_file[i]
                image_extent = seamline_file[i+1]
                image_bbox = get_image_bbox(image_extent)
                if is_image_in_area(image_data, image_bbox, area_outline, available_images):
                    buildings_in_image = buildings.get_buildings_in_area(image_bbox)
                    process_image(images_path, image_data, buildings_in_image, config)

if __name__ == '__main__':
    create_dataset('')

        # exit()
        # t = default_timer()
        # image_name = path.basename(image_path).split('.')[0]
        # print('Processing ', image_name, '...')
        # image = Image.open(image_path)
        
        # tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(tile_size, tile_size))

        # print('Exporting semantic segmentation', end=' - ')
        # t = default_timer()
        # tiled_image.export_semantic_segmentation(annotation_format, label_walls)
        # print(default_timer() -t, 'seconds')
