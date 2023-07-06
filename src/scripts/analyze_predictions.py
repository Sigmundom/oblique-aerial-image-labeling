import json
import math
import os
import click
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import box
from tqdm import tqdm
from config import BUFFER, HEIGHT_PX_OFFSET, HEIGHT_RASTER_SIZE, PX_P_M, RASTER_SIZE, THRESH, TILE_SIZE_M
from core.image_data import ImageDataList, ImageDataRecord
from utils import ensure_folder_exists, filter_height_values, get_heights_tiff, Camera, save_corrected_nadir_mask, make_histogram, save_image
from PIL import Image, ImageTransform
from scipy.ndimage import median_filter, gaussian_filter
from scipy.interpolate import LinearNDInterpolator
import rasterio
from rasterio.transform import from_bounds

import rasterio
from rasterio.transform import  rowcol

from utils.convolution import smooth
# from utils.camera import Camera
# from utils.nadir_mask import save_corrected_nadir_mask
# from utils.convolution import smooth
# from utils.ensure_folder_exisits import ensure_folder_exists, make_histogram








def interpolate_heights(height_values, building_pixels, height, width):
    z_grid = np.zeros((height, width))

    interp_func = LinearNDInterpolator(height_values[:,:2], height_values[:,2], fill_value=0)

    res = interp_func(*building_pixels.T)

    z_grid[building_pixels[:,0], building_pixels[:, 1]] = res

    return z_grid

def analyze_tile(tile_dir: str, image_data: list[ImageDataRecord]):
    mask_dir = os.path.join(tile_dir, 'masks')
    mask_paths = [os.path.join(mask_dir,mask) for mask in os.listdir(mask_dir)]
    masks = [plt.imread(mask) for mask in mask_paths]
    assert len(masks[0].shape) == 2
    
    # Check if any of the masks has detected buildings
    tile_contains_buildings = False
    for mask in masks:
        if np.sum(mask>THRESH) > 0.001:
            tile_contains_buildings = True
    if not tile_contains_buildings: return #TODO: Possibly create a empty tile?

    result_dir = os.path.join(tile_dir, 'results')
    ensure_folder_exists(result_dir)

    with open(os.path.join(tile_dir, 'info.json'), 'r') as f:
        tile_info = json.load(f)

    tile_x = tile_info['x']
    tile_y = tile_info['y']
    tile_maxx = tile_x + TILE_SIZE_M
    tile_maxy = tile_y + TILE_SIZE_M

    aoi = box(tile_x, tile_y, tile_maxx, tile_maxy)
    

    heights_tiff = get_heights_tiff(aoi.buffer(BUFFER), tile_size=HEIGHT_RASTER_SIZE)
    heights = heights_tiff.read(1)

    # # Save image of heights just because
    h = ((heights / heights.max()) * 255).astype(np.uint8)
    plt.imsave(os.path.join(result_dir, 'heights.png'), h)
    


    x_values = np.linspace(tile_x-BUFFER+HEIGHT_PX_OFFSET, tile_x + BUFFER + TILE_SIZE_M-HEIGHT_PX_OFFSET, HEIGHT_RASTER_SIZE)
    y_values = np.linspace(tile_y-BUFFER+HEIGHT_PX_OFFSET, tile_y + BUFFER + TILE_SIZE_M-HEIGHT_PX_OFFSET, HEIGHT_RASTER_SIZE)
    x = np.tile(x_values, HEIGHT_RASTER_SIZE) # [1,2,3,1,2,3,1,...]
    y = np.repeat(y_values[::-1], HEIGHT_RASTER_SIZE) #[3,3,3,2,2,2,1,...]
    xyz = np.array([x, y, heights.flatten()]).T # List of each coordinate, heights included
    

    result = np.zeros((RASTER_SIZE, RASTER_SIZE))
    # count = np.zeros((RASTER_SIZE,RASTER_SIZE), dtype=np.uint8)

    transform = from_bounds(tile_x, tile_y, tile_maxx, tile_maxy, RASTER_SIZE, RASTER_SIZE)
    profile = heights_tiff.profile
    profile['dtype'] = rasterio.uint8
    profile['width'] = RASTER_SIZE
    profile['height'] = RASTER_SIZE
    profile['transform'] = transform
    image_info:dict = tile_info['image_info']
    for cam_id, info in image_info.items():
        im_name = info['image_name']
        im_data = next(x for x in image_data if x.name == im_name)

        cropbox_size = info['cropbox_size']
        dx = info['dx']
        dy = info['dy']
        minx, miny, maxx, maxy = info['bbox_ic']

        mask_w = int(cropbox_size - 2*dx)
        mask_h = int(cropbox_size - 2*dy)
        cropbox = (math.floor(dx), math.floor(dy), cropbox_size - math.ceil(dx), cropbox_size - math.ceil(dy))
        mask_im = Image.open(os.path.join(mask_dir, f'{im_name}.png')).resize((cropbox_size, cropbox_size))
        
        if cam_id == 'Cam0N':
            nadir_result = save_corrected_nadir_mask(tile_x, tile_y, result_dir, heights, heights_tiff.transform, im_data, minx, miny, mask_im, mask_h)
            nadir_result[nadir_result < THRESH] = 0
            result += nadir_result**2
        else:
            mask = np.array(mask_im.crop(cropbox))
            mask = mask / 255
            
            if not np.any(mask > THRESH): continue
            building_pixels = np.argwhere(mask > THRESH)

            laser_data_ic = im_data.wc_to_ic(xyz)
            laser_data_tc = np.zeros((HEIGHT_RASTER_SIZE*HEIGHT_RASTER_SIZE, 3), dtype=np.float32)
            laser_data_tc[:,0] = maxy - laser_data_ic[:,1] 
            laser_data_tc[:,1] = laser_data_ic[:,0] - minx
            laser_data_tc[:,2] = xyz[:,2]

            x_mask = np.equal.outer(laser_data_tc[:,0].astype(np.int16), building_pixels[:, 0])
            y_mask = np.equal.outer(laser_data_tc[:,1].astype(np.int16), building_pixels[:, 1])
            selected_indices = np.where(np.logical_and(x_mask, y_mask).any(axis=1))[0]

            # Select points using the indices
            laser_data_tc = laser_data_tc[selected_indices]
            laser_data_tc = filter_height_values(laser_data_tc, mask_w, mask_h, cropbox, cropbox_size)

            if len(laser_data_tc) < 4: continue

            z_grid = interpolate_heights(laser_data_tc, building_pixels, mask_h, mask_w)
            z = ((z_grid / z_grid.max()) * 255).astype(np.uint8)
            plt.imsave(os.path.join(result_dir, f'z_grid_{cam_id}.png'), z)

            rows = building_pixels[:,0]
            cols = building_pixels[:,1]
            x = cols + minx
            y = maxy - rows
            Z = z_grid[rows, cols]

            lng, lat = im_data.ic_to_wc(x, y, Z) 
            

            result_rows, result_cols = rowcol(transform, lng, lat)
            tmp_result = np.zeros((RASTER_SIZE, RASTER_SIZE))
            for r, c, score  in zip(result_rows, result_cols, mask[rows, cols]):
                if r < 0 or r > RASTER_SIZE-1 or c < 0 or c > RASTER_SIZE-1:
                    continue
                else:
                    tmp_result[r,c] = max(tmp_result[r,c], score)
                    # count[r,c] += 1
            result += tmp_result
            # tmp_result[tmp_result>5] = 5.0
            save_image(tmp_result, f'{result_dir}/result_{cam_id}.png')

    plt.close()
    # plt.imsave(f'{result_dir}/count.png', count)
    
    # result_smooth = median_filter(result, 5)
    result_smooth = gaussian_filter(result, 5)
    # result_smooth = smooth(result, count)
    result_sharp_5 = result_smooth >= 1.5
    result_sharp_75 = result_smooth >= 1.75
    result_sharp_2 = result_smooth >= 2
    result_sharp_2_25 = result_smooth >= 2.25
    
    # result[result>5] = 5
    save_image(result, f'{result_dir}/result.png')
    save_image(result_smooth, f'{result_dir}/result_smooth.png')
    save_image(result_sharp_5, f'{result_dir}/result_sharp_1_5.png', mode='1')
    save_image(result_sharp_75, f'{result_dir}/result_sharp_1_75.png', mode='1')
    save_image(result_sharp_2, f'{result_dir}/result_sharp_2.png', mode='1')
    save_image(result_sharp_2_25, f'{result_dir}/result_sharp_2_25.png', mode='1')


def compile_tiles(analysis_folder):
    output_folder = os.path.join(analysis_folder, 'compiled')
    ensure_folder_exists(output_folder)
    tile_xs = []
    tile_ys = []
    for tile_folder in [f.name for f in os.scandir(analysis_folder) if f.is_dir() and '_' in f.name]:
        x, y = tile_folder.split('_')
        tile_xs.append(int(x))
        tile_ys.append(int(y))
    tile_minx, tile_maxx = min(tile_xs), max(tile_xs)
    tile_miny, tile_maxy = min(tile_ys), max(tile_ys)
    area_width = tile_maxx - tile_minx + TILE_SIZE_M
    area_height = tile_maxy - tile_miny + TILE_SIZE_M

    nadir_result = np.zeros((area_height*PX_P_M, area_width*PX_P_M), dtype=np.uint8)
    result = np.zeros_like(nadir_result)

    nadir_result_sharp = np.zeros_like(nadir_result, dtype=bool)
    result_1_5 = np.zeros_like(nadir_result, dtype=bool)
    result_1_75 = np.zeros_like(nadir_result, dtype=bool)
    result_2 = np.zeros_like(nadir_result, dtype=bool)
    result_2_25 = np.zeros_like(nadir_result, dtype=bool)


    for tile_folder in tqdm([f.name for f in os.scandir(analysis_folder) if f.is_dir and '_' in f.name]):
        if not os.path.exists(os.path.join(analysis_folder, tile_folder, 'results')): continue

        x, y = tile_folder.split('_')
        x = (int(x) - tile_minx) * PX_P_M
        y = (tile_maxy - int(y)) * PX_P_M

        nadir_im = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'nadir_result.png')), dtype=np.uint8)
        nadir_result[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = nadir_im

        nadir_sharp = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'nadir_result_sharp.png')), dtype=bool)
        nadir_result_sharp[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = nadir_sharp

        combined_im = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'result_smooth.png')), dtype=np.uint8)
        result[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = combined_im

        combined_im_1_5 = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'result_sharp_1_5.png')), dtype=bool)
        result_1_5[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = combined_im_1_5
  
        combined_im_1_75 = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'result_sharp_1_75.png')), dtype=bool)
        result_1_75[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = combined_im_1_75

        combined_im_2 = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'result_sharp_2.png')), dtype=bool)
        result_2[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = combined_im_2

        combined_im_2_25 = np.array(Image.open(os.path.join(analysis_folder, tile_folder, 'results', 'result_sharp_2_25.png')), dtype=bool)
        result_2_25[y:y+RASTER_SIZE, x:x+RASTER_SIZE] = combined_im_2_25

    save_image(nadir_result, f'{output_folder}/nadir.png')
    save_image(result, f'{output_folder}/combined.png')
    save_image(nadir_result_sharp, f'{output_folder}/nadir_sharp.png', mode='1')
    save_image(result_1_5, f'{output_folder}/combined_1_5.png', mode='1')
    save_image(result_1_75, f'{output_folder}/combined_1_75.png', mode='1')
    save_image(result_2, f'{output_folder}/combined_2.png', mode='1')
    save_image(result_2_25, f'{output_folder}/combined_2_25.png', mode='1')

    transform = from_bounds(tile_minx, tile_miny, tile_maxx, tile_maxy, area_width, area_height)

    combined_prob = (result / 255).astype(np.float32)
    nadir_prob = (nadir_result / 255).astype(np.float32)

    with rasterio.open(f'{output_folder}/combined.tiff', 'w', driver='GTiff', height=area_height*PX_P_M, width=area_width*PX_P_M, count=1, dtype=rasterio.float32,
                   crs='EPSG:25832', transform=transform) as dst:
        dst.write(combined_prob, 1)

    with rasterio.open(f'{output_folder}/nadir.tiff', 'w', driver='GTiff', height=area_height*PX_P_M, width=area_width*PX_P_M, count=1, dtype=rasterio.float32,
                   crs='EPSG:25832', transform=transform) as dst:
        dst.write(nadir_prob, 1)


@click.command()
@click.argument('config')
def analyze_predictions(config):
    #### Parameters ####
    # config = 'config/analysis/grimstad.json'
    # config = 'config/analysis/lindesnes.json'
    ###################
    with open(config, encoding='utf8') as f:
        config = json.load(f)

    analysis_folder = config['folder']

    compile_tiles(analysis_folder)
    exit()

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

    
    for tile_folder in tqdm([f.path for f in os.scandir(analysis_folder) if f.is_dir()]):
        
        analyze_tile(tile_folder, image_data)

    print('Compiling...')
    compile_tiles(analysis_folder)



if __name__ == '__main__':
    # masks_dir = 'test/masks'
    # x = 476155.42
    # y = 6465926.15
    # d = 25
    # aoi = box(x-d, y-d, x+d, y+d)
    analyze_predictions()