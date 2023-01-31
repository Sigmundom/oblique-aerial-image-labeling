import click
from pycocotools.coco import COCO
from os import path

@click.command()
@click.argument('coco_path')
@click.option('-o', '--output-folder', default='outputs')
@click.option('-i', '--image-folder', default=None)
def main(coco_path, output_folder, image_folder):
    click.echo(coco_path)
    coco = COCO(coco_path)
    if image_folder == None:
        image_folder = path.join(*coco_path.split('/')[:-2], 'images')
    for img_id, annotation in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        
        # Export image
        
        # print(img, annotations[0]['id'])
