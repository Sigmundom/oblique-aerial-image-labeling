import gc
import json
import os
import click
from tqdm import tqdm

from core import AnnotatedTiledImage, WCBuildingCollection, ImageDataList
from utils import Camera
from PIL import Image
from shapely.geometry import shape

Image.MAX_IMAGE_PIXELS = 287944704

@click.command()
@click.option('-c', '--config', default='data/config/lindesnes.json')
def create_dataset(config):
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
    
    for area in config['areas']:
        print('Creating building collection')
        buildings = WCBuildingCollection(area['cityjson'], area['municipality'])
        area_outline = buildings.outline
        print('Complete')
        
        exclude_area = area['exclude'] if "exclude" in area else None
        include_area = area['include'] if 'include' in area else None
        i = 0
        for image in tqdm(image_data):
            if  image.is_image_in_area(area_outline):
                buildings_in_image = buildings.get_buildings_in_area(image.bbox)
                tiled_image = AnnotatedTiledImage(buildings_in_image, image, output_folder=config['output_folder'], tile_size=config['tile_size'], exclude_area=exclude_area, include_area=include_area)
                annotation_format = config['annotation_format']
                label_walls = config['label_walls']
                tiled_image.export_semantic_segmentation(annotation_format, label_walls)
                del tiled_image
                del buildings_in_image
                i+=1
                gc.collect()   

if __name__ == '__main__':
    create_dataset()
