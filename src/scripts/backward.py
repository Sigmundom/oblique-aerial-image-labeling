import json
import os
import re
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import box
from core.image_data import ImageDataList
from utils import get_laser_data
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
from scipy.interpolate import LinearNDInterpolator

from utils.camera import Camera
import rasterio
from rasterio.transform import xy, rowcol

from utils.convolution import smooth


def interpolate_heights(points, width, height):
    z_grid = -np.ones((height, width))
    for point in points:
        r, c = np.floor(point[:2]).astype(int)
        if 0 > c or c >= width or 0 > r or r >= height: continue
        z = point[2]
        if z > z_grid[r, c]:
            z_grid[r,c] = z

    interp_func = LinearNDInterpolator(np.argwhere(z_grid != -1), z_grid[z_grid != -1], fill_value=0)
    missing_indices = np.argwhere(z_grid == -1)

    z_grid[missing_indices[:, 0], missing_indices[:, 1]] = interp_func(*missing_indices.T)

    z_grid = median_filter(z_grid, size=(15,5))

    return z_grid

def backward(config, masks_dir, aoi):
    with open(config, encoding='utf8') as f:
        config = json.load(f)

    cameras = {camera_info['cam_id'].lower(): Camera(camera_info) for camera_info in config['cameras']}
    image_folder = config['images'][0]
    image_data_paths = config['image_data']
    masks = {}
    for file_name in os.listdir(masks_dir):
        split_i = re.search('Cam\d[A-Z]', file_name).end()
        image_name=  file_name[:split_i]
        masks[image_name] = {
            'bbox': [float(x) for x in file_name[split_i+1:-8].split('_')],
            'path': os.path.join(masks_dir, file_name)
        }

    image_paths = [os.path.join(image_folder, f'{im_name}.jpg') for im_name in masks.keys()]

    image_data_format = config['image_data_format']
    if image_data_format == 'sos':
        image_data_list = ImageDataList.from_sos(image_paths, image_data_paths, cameras)
    elif image_data_format == 'dbf':
        image_data_list = ImageDataList.from_shp(image_paths, image_data_paths, cameras)
    else:
        raise ValueError(f'"{image_data_format}" is not a valid format. Must be "sos" or "dbf"')
    
    laser_data = get_laser_data(aoi)
    z_values = laser_data.read(1)
    h = z_values / z_values.max()
    h = h * 255
    h = h.astype(np.uint8)
    plt.imsave('outputs/test/heights.png', h)
    


    rows = np.arange(512).repeat(512)
    cols = np.resize(np.arange(512), 512*512)
    x, y = xy(laser_data.transform, rows, cols)
    xyz = np.array([x, y, z_values.flatten()]).T

    result = np.zeros((512, 512))
    count = np.zeros((512,512), dtype=np.uint8)
    profile = laser_data.profile
    profile['dtype'] = rasterio.uint8
    output = rasterio.open("outputs/test/output.tif", "w", **profile)

    for im_data in image_data_list:
        mask_dict = masks[im_data.name]
        laser_data_ic = im_data.wc_to_ic(xyz)
        minx, miny, maxx, maxy = mask_dict['bbox']
        mask_w = round(maxx-minx)
        mask_h = round(maxy-miny)
        laser_data_tc = np.zeros((512*512, 3), dtype=np.float32)
        laser_data_tc[:,0] = maxy - laser_data_ic[:,1] 
        laser_data_tc[:,1] = laser_data_ic[:,0] - minx
        laser_data_tc[:,2] = xyz[:,2]

        z_grid = interpolate_heights(laser_data_tc, mask_w, mask_h)
        plt.imsave(f'test/z_grids_median/{im_data.name}.png', z_grid)

        mask = np.array(Image.open(mask_dict['path']).resize((mask_w, mask_h)))
        mask = mask / mask.max()
        indices = np.where(mask < 2)
        x = indices[1] + minx
        y = maxy - indices[0]
        Z = z_grid[indices]

        lng, lat = im_data.ic_to_wc(x, y, Z) 
        rows, cols = rowcol(output.transform, lng, lat)
        for r, c, score in zip(rows, cols, mask[indices]):
            if r < 0 or r > 511 or c < 0 or c > 511:
                continue
            else:
                result[r,c] += score
                count[r,c] += 1

    plt.imsave('outputs/test/count.png', count)
    count_0 = (count == 0).astype(np.uint8) * 255
    plt.imsave('outputs/test/count_0.png', count_0)

    result_1 = result / result.max()
    result_255 = result_1 * 255
    result_255 = result_255.astype(np.uint8)
    output.write(result_255, 1)
    plt.imsave('outputs/test/output.png', result_255)

    # count += 1
    result_scaled = result/(count+1)
    result_scaled = result_scaled / result_scaled.max()
    result_scaled_255 = (result_scaled * 255).astype(np.uint8)
    plt.imsave('outputs/test/output_scaled.png', result_scaled_255)

    # weights = count.astype(np.float64) / count.sum()
    # result_smooth = gaussian_filter(result, radius=3)

    smoothed = smooth(result_scaled, count)
    # Print the smoothed raster
    # smoothed = smoothed_sum / smoothed_sum.max()
    # plt.imsave('outputs/test/smooth.png', (smoothed*255).astype(np.uint8))
    im = Image.fromarray((smoothed*255).astype(np.uint8))
    im.save('outputs/test/smooth.png')
    
    print(smoothed.max())
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1, 1.1]:
        save(smoothed, thresh)


def save(result, thresh):
    im = Image.fromarray((result>=thresh).astype(np.uint8)*255)
    im.save(f'outputs/test/output_smooth_{thresh}.png')
    # plt.imsave(f'outputs/test/output_smooth_{thresh}.png', (result>=thresh).astype(np.uint8)*255)


if __name__ == '__main__':
    config = 'data/config/grimstad_new.json'
    masks_dir = 'test/masks'
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    backward(config, masks_dir, aoi)