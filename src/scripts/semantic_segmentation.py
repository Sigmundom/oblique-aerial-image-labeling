import json
from os import path
from timeit import default_timer
import click
from PIL import Image

from core import AnnotatedTiledImage
from utils import get_image_data



@click.command()
@click.argument('image_path')
@click.argument('seamline_path')
@click.option('-o', '--output-folder', default='outputs')
@click.option('-ts', '--tile-size', default=512) 
@click.option('-f','--annotation-format', default='mask', type=click.Choice(['mask', 'coco'], case_sensitive=False))
@click.option('-w', '--label-walls', is_flag=True)
def semantic_segmentation(image_path, seamline_path, output_folder, tile_size, annotation_format, label_walls):
    t_start = default_timer()
    print('Loading CityGML', end='')
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        cityjson = json.load(f)
    print(' - Complete - Took', default_timer() - t_start, 'seconds' )

    t = default_timer()
    image_name = path.basename(image_path).split('.')[0]
    print('Processing ', image_name, '...')
    image = Image.open(image_path)
    image_data = get_image_data(seamline_path, image_name)
    
    tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(tile_size, tile_size))

    print('Exporting semantic segmentation', end=' - ')
    t = default_timer()
    tiled_image.export_semantic_segmentation(annotation_format, label_walls)
    print(default_timer() -t, 'seconds')
