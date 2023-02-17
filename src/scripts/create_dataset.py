import json
from typing import OrderedDict
import numpy as np
from os import path
import os
from timeit import default_timer
import click
from PIL import Image
import shapely.geometry as sg

from core import AnnotatedTiledImage, WCBuildingCollection
from libs.sosi import read_sos

import urllib.request

def get_municipality_border(name: str, number: int): 
    file_path = f'/data/municipalities/{number}-{name}.json'
    if os.path.exists(file_path):
        with open(file_path, encoding='utf8') as f:
            geojson = json.load(f)
    else:
        response = urllib.request.urlopen(f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{number}/omrade', ).read()
        geojson = json.loads(response)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(geojson, f)
    
    return sg.Polygon(geojson['omrade']['coordinates'][0][0])

def is_image_in_area(image_data: dict, image_extent: OrderedDict, area: sg.Polygon, available_images: list[str]) -> bool:
    if not "ImageName" in image_data: return False
                
    image_name = image_data["ImageName"]
    if not f'{image_name}.jpg' in available_images: return False

    image_outline = np.array(list(image_extent.values())[0], dtype=np.float64)
    image_outline[:, [1, 0]] = image_outline[:, [0, 1]]

    image_outline /= 100 # Coordinates are given in cm for some reason. Convert to m.
    image_outline = sg.Polygon(image_outline)
    intersection = image_outline.intersection(area)
    overlap = intersection.area / image_outline.area
    if overlap > 0.99:
        return True
    elif overlap > 0.5:
        print('Partial overlap:', overlap)
        return False

def process_image(images_path, image_data, cityjson, config):
    image_name = image_data['ImageName']

    print('Processing ', image_name, '...')
    image = Image.open(path.join(images_path, f'{image_name}.jpg'))
    tile_size = config['tile_size']
    tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=config['output_folder'], tile_size=(tile_size, tile_size))

    print('Exporting semantic segmentation', end=' - ')
    t = default_timer()
    annotation_format = config['annotation_format']
    label_walls = config['label_walls']
    tiled_image.export_semantic_segmentation(annotation_format, label_walls)
    print(default_timer() -t, 'seconds')

@click.command()
@click.option('-c', '--config', default='data/config/grimstad.json')
# @click.option('-o', '--output-file', default='data/data.json')
@click.option('-ts', '--tile-size', default=512) 
@click.option('-f','--annotation-format', default='mask', type=click.Choice(['mask', 'coco'], case_sensitive=False))
@click.option('-w', '--label-walls', is_flag=True)
def create_dataset(config, tile_size, annotation_format, label_walls):
    with open(config, encoding='utf8') as f:
        config = json.load(f)
    for area in config['areas']:
        municipality_border = get_municipality_border(area['name'], area['municipality_number'])
 
        for data_source in config['data']:
            seamline = data_source['seamline']
            images_path = data_source['images']
            available_images = os.listdir(images_path)
            seamline_file = list(read_sos(seamline).values())
            # images = []
            for i in range(1,len(seamline_file),2):
                image_data = seamline_file[i]
                image_extent = seamline_file[i+1]
                if is_image_in_area(image_data, image_extent, municipality_border, available_images):
                    process_image()

               

        exit()
        t = default_timer()
        image_name = path.basename(image_path).split('.')[0]
        print('Processing ', image_name, '...')
        image = Image.open(image_path)
        
        tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(tile_size, tile_size))

        print('Exporting semantic segmentation', end=' - ')
        t = default_timer()
        tiled_image.export_semantic_segmentation(annotation_format, label_walls)
        print(default_timer() -t, 'seconds')
