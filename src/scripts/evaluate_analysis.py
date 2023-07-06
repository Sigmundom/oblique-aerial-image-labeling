from PIL import Image
import click
from matplotlib import pyplot as plt
import numpy as np

from utils import ensure_folder_exists, save_image

@click.command()
@click.argument('folder')
def evaluate_analysis(folder):
    input_folder = f'{folder}/compiled'
    nadir_path = f'{input_folder}/nadir_sharp.png'
    combined_path = f'{input_folder}/combined_2.png'
    ground_truth = 'ground_truth.png'

    nadir = np.array(Image.open(nadir_path))
    combined = np.array(Image.open(combined_path))
    truth = np.array(Image.open(ground_truth).convert('1'))
    print(nadir)
    print(truth)
    print(nadir.shape)
    print(truth.shape)

    intersection = nadir & combined
    union = nadir | combined

    # not_detected = truth & np.logical_not(nadir | combined)

    only_nadir = nadir & np.logical_not(combined)

    only_combined = combined & np.logical_not(nadir)

    opacity = 96

    im = np.zeros((*nadir.shape, 4), dtype=np.uint8)
    im[intersection & truth,:] = [255,255,255,opacity]
    im[intersection & np.logical_not(truth)] = [0, 255, 0, opacity]
    im[truth & np.logical_not(union)] = [255, 0, 0, opacity]
    im[only_combined & truth] = [255, 255, 0, opacity]
    im[only_combined & np.logical_not(truth)] = [128, 0, 128, opacity]
    im[only_nadir & truth] = [0, 0, 255, opacity]
    im[only_nadir & np.logical_not(truth)] = [255, 128, 0, opacity]

    output_folder = f'{folder}/evaluate'
    ensure_folder_exists(output_folder)
    # plt.imsave(f'{output_folder}/evaluate.png', im)


    background = Image.open(f'{folder}/test_area.jpeg')
    im = Image.fromarray(im)

    hd = background.resize(im.size)
    hd.paste(im, (0,0), im)
    hd.save(f'{output_folder}/evaluate_hd.png')

    im = im.resize(background.size)
    print('HEy 1')
    background.paste(im, (0,0), im)
    print('Hey 2')
    background.save(f'{output_folder}/evaluate.png')
    print('success')
    exit()
    save_image(background, f'{output_folder}/evaluate.png')
    save_image(intersection, f'{output_folder}/intersection.png', '1')
    save_image(only_nadir, f'{output_folder}/only_nadir.png', '1')
    save_image(only_combined, f'{output_folder}/only_combined.png', '1')


if __name__ == '__main__':
    evaluate_analysis()