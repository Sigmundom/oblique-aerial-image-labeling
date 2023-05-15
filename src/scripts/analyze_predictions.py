from itertools import chain
import json
import math
import os
import click
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import box
from core.image_data import ImageDataList, ImageDataRecord
from scripts.prepare_analysis import TILE_SIZE_M, TILE_SIZE_PX
from utils import get_heights_tiff
from PIL import Image, ImageTransform
from scipy.ndimage import median_filter
from scipy.interpolate import LinearNDInterpolator

from utils.camera import Camera
import rasterio
from rasterio.transform import xy, rowcol

from utils.convolution import smooth
from utils.utils import ensure_folder_exists

RESULT_TILE_SIZE_PX = 500

# def save(result, thresh):
#     im = Image.fromarray((result>=thresh).astype(np.uint8)*255)
#     im.save(f'outputs/test/output_smooth_{thresh}.png')
    # plt.imsave(f'outputs/test/output_smooth_{thresh}.png', (result>=thresh).astype(np.uint8)*255)


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


def analyze_tile(tile_dir: str, image_data: list[ImageDataRecord]):
    mask_dir = os.path.join(tile_dir, 'masks')
    mask_paths = [os.path.join(mask_dir,mask) for mask in os.listdir(mask_dir)]
    masks = [plt.imread(mask) for mask in mask_paths]
    assert len(masks[0].shape) == 2
    
    # Check if any of the masks has detected buildings
    tile_contains_buildings = False
    for mask in masks:
        if np.sum(mask>0.5) > 0.001:
            tile_contains_buildings = True
    if not tile_contains_buildings: return #TODO: Possibly create a empty tile?

    with open(os.path.join(tile_dir, 'info.json'), 'r') as f:
        tile_info = json.load(f)

    tile_x = tile_info['x']
    tile_y = tile_info['y']
    print(tile_x, tile_y)

    aoi = box(tile_x, tile_y, tile_x+TILE_SIZE_M, tile_y+TILE_SIZE_M)
    heights_tiff = get_heights_tiff(aoi)
    
    result_dir = os.path.join(tile_dir, 'results')
    ensure_folder_exists(result_dir)

    heights = heights_tiff.read(1)

    # Save image of heights just because
    h = ((heights / heights.max()) * 255).astype(np.uint8)
    plt.imsave(os.path.join(result_dir, 'heights.png'), h)
    

    rows = np.arange(512).repeat(512)
    cols = np.resize(np.arange(512), 512*512)
    x, y = xy(heights_tiff.transform, rows, cols) #World coordinate for each pixel
    xyz = np.array([x, y, heights.flatten()]).T # List of each coordinate, heights included

    result = np.zeros((512, 512))
    count = np.zeros((512,512), dtype=np.uint8)
    profile = heights_tiff.profile
    profile['dtype'] = rasterio.uint8
    output = rasterio.open(f"{result_dir}/output.tif", "w", **profile)

    image_info:dict = tile_info['image_info']
    for cam_id, info in image_info.items():
        im_name = info['image_name']
        im_data = next(x for x in image_data if x.name == im_name)

        laser_data_ic = im_data.wc_to_ic(xyz)
        minx, miny, maxx, maxy = info['bbox_ic']

        laser_data_tc = np.zeros((512*512, 3), dtype=np.float32)
        laser_data_tc[:,0] = maxy - laser_data_ic[:,1] 
        laser_data_tc[:,1] = laser_data_ic[:,0] - minx
        laser_data_tc[:,2] = xyz[:,2]

        cropbox_size = info['cropbox_size']
        dx = info['dx']
        dy = info['dy']
        mask_w = int(cropbox_size - 2*dx)
        mask_h = int(cropbox_size - 2*dy)
        z_grid = interpolate_heights(laser_data_tc, mask_w, mask_h)
        plt.imsave(f'{result_dir}/z_grid.png', z_grid)

        cropbox = (math.floor(dx), math.floor(dy), cropbox_size - math.ceil(dx), cropbox_size - math.ceil(dy))
        mask_im = Image.open(os.path.join(mask_dir, f'{im_name}.png')).resize((cropbox_size, cropbox_size))
        mask = np.array(mask_im.crop(cropbox))
        mask = mask / 255
        indices = np.where(mask < 2)
        print(indices)
        x = indices[1] + minx
        y = maxy - indices[0]
        Z = z_grid[indices]

        if cam_id == 'Cam0N':
            # Transforms and saves the nadir mask with correct orientation 
            left, right = tile_x, tile_x + TILE_SIZE_M
            bottom, top = tile_y, tile_y + TILE_SIZE_M
            corners_lng = [left, left, right, right]
            corners_lat = [top, bottom, bottom, top]

            corners_row, corners_col = rowcol(output.transform, corners_lng, corners_lat)
            for arr in [corners_row, corners_col]:
                for i in range(4):
                    if arr[i] >= TILE_SIZE_PX:
                        arr[i] = TILE_SIZE_PX-1
                    elif arr[i] < 0:
                        arr[i] = 0
            
            corners_xy = im_data.wc_to_ic(np.array([corners_lng, corners_lat, heights[corners_row, corners_col]]).T)
            corners_x = corners_xy[:,0]
            corners_y = corners_xy[:,1]
            corners_x -= minx
            corners_y = mask_h - (corners_y - miny)
            
            # Define 8-tuple with x,y coordinates of top-left, bottom-left, bottom-right and top-right corners and apply
            transform = np.empty(8)
            transform[0::2] = corners_x
            transform[1::2] = corners_y
            nadir_result = mask_im.transform((TILE_SIZE_PX, TILE_SIZE_PX), ImageTransform.QuadTransform(transform))
            nadir_result_arr = np.array(nadir_result) / 255
            nadir_result_sharp = Image.fromarray((nadir_result_arr > 0.5).astype(np.uint8)*255)
            
            nadir_result.save(f'{result_dir}/nadir_result.png')
            nadir_result_sharp.save(f'{result_dir}/nadir_result_sharp.png')

        corners_lng, corners_lat = im_data.ic_to_wc(x, y, Z) 
        rows, cols = rowcol(output.transform, corners_lng, corners_lat)
        for r, c, score in zip(rows, cols, mask[indices]):
            if r < 0 or r > 511 or c < 0 or c > 511:
                continue
            else:
                result[r,c] += score
                count[r,c] += 1

    # plt.imsave(f'{result_folder}/count.png', count)
    # count_0 = (count == 0).astype(np.uint8) * 255
    # plt.imsave(f'{result_folder}/count_0.png', count_0)

    # result_1 = result / result.max()
    # result_255 = result_1 * 255
    # result_255 = result_255.astype(np.uint8)
    # output.write(result_255, 1)
    # plt.imsave(f'{result_folder}/output.png', result_255)

    result_scaled = result/(count+1e-15)
    # result_scaled = result_scaled / result_scaled.max()
    result_scaled_im = Image.fromarray((result_scaled * 255).astype(np.uint8))
    result_scaled_im.save(f'{result_dir}/output_scaled.png')

    smoothed = smooth(result_scaled, count)
    # Print the smoothed raster
    # smoothed = smoothed_sum / smoothed_sum.max()
    # plt.imsave(f'{result_folder}/smooth.png', (smoothed*255).astype(np.uint8))
    im = Image.fromarray((smoothed*255).astype(np.uint8))
    im.save(f'{result_dir}/smooth.png')
    
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        im = Image.fromarray((smoothed>=thresh).astype(np.uint8)*255)
        im.save(f'{result_dir}/thresh_{thresh}.png')




@click.command()
def analyze_predictions():
    #### Parameters ####
    config = 'config/analysis/lindesnes.json'
    ###################

    with open(config, encoding='utf8') as f:
        config = json.load(f)

    analysis_folder = config['folder']
    cameras = {camera_info['cam_id']: Camera(camera_info) for camera_info in config['cameras']}

    # Find path to all images used in analysis
    image_folder = config['images'][0]
    # regex = r'Cam[04567][BNRFL]'
    # re.search(regex, file).end()
    image_paths = [os.path.join(image_folder, file[:-4]) for _, _, files in os.walk(analysis_folder) for file in files]
    
    image_data_paths = config['image_data']
    image_data_format = config['image_data_format']
    if image_data_format == 'sos':
        image_data = ImageDataList.from_sos(image_paths, image_data_paths, cameras)
    elif image_data_format == 'shp':
        image_data = ImageDataList.from_shp(image_paths, image_data_paths, cameras)
    else:
        raise ValueError(f'"{image_data_format}" is not a valid format. Must be "sos" or "shp"')

    
    for tile_folder in os.listdir(analysis_folder):
        
        analyze_tile(os.path.join(analysis_folder, tile_folder), image_data)

    


if __name__ == '__main__':
    # masks_dir = 'test/masks'
    # x = 476155.42
    # y = 6465926.15
    # d = 25
    # aoi = box(x-d, y-d, x+d, y+d)
    analyze_predictions()