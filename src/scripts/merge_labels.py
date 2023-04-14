import glob
from os import path
import click
from matplotlib import pyplot as plt

from tqdm import tqdm

@click.command()
@click.argument('label_folder')
def merge_labels(label_folder):
    for im_path in tqdm(glob.glob(path.join(label_folder, '*.png'))):
        im = plt.imread(im_path)
        im[im>0] = 1
        plt.imsave(im_path.replace('label_walls', 'label'), im)

if __name__ == '__main__':
    merge_labels()
