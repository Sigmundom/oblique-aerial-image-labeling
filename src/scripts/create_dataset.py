import json
import os
import click
from tqdm import tqdm

from core import AnnotatedTiledImage, WCBuildingCollection, ImageData
from utils import Camera
from PIL import Image

Image.MAX_IMAGE_PIXELS = 287944704


@click.command()
@click.option('-c', '--config', default='data/config/lindesnes.json')
def create_dataset(config):
    # tracemalloc.start()
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
    
    for area in config['areas']:
        print('Creating building collection')
        buildings = WCBuildingCollection(area['cityjson'], area['municipality'])
        area_outline = buildings.outline
        print('Complete')
        i = 0
        # old_snapshot = tracemalloc.take_snapshot()
        for image in tqdm(image_data):
            if  image.is_image_in_area(area_outline):
                buildings_in_image = buildings.get_buildings_in_area(image.bbox)
                tiled_image = AnnotatedTiledImage(buildings_in_image, image, output_folder=config['output_folder'], tile_size=config['tile_size'])
                annotation_format = config['annotation_format']
                label_walls = config['label_walls']
                tiled_image.export_semantic_segmentation(annotation_format, label_walls)
                del tiled_image
                i+=1
            # if i % 10 == 9:
            #     new_snapshot = tracemalloc.take_snapshot()
            #     top_stats = new_snapshot.compare_to(old_snapshot, 'lineno')

            #     print("[ Top 10 ]")
            #     for stat in top_stats[:10]:
            #         print(stat)
            #     old_snapshot = new_snapshot     


if __name__ == '__main__':
    create_dataset()

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
