import numpy as np
from PIL import Image, ImageTransform
from rasterio.transform import rowcol

from config import RASTER_SIZE, THRESH, TILE_SIZE_M


def save_corrected_nadir_mask(tile_x, tile_y, result_dir, heights, transform, im_data, minx, miny, mask_im, mask_h):
    left, right = tile_x, tile_x + TILE_SIZE_M
    bottom, top = tile_y, tile_y + TILE_SIZE_M
    corners_lng = [left, left, right, right]
    corners_lat = [top, bottom, bottom, top]

    corners_row, corners_col = rowcol(transform, corners_lng, corners_lat)
    for arr in [corners_row, corners_col]:
        for i in range(4):
            if arr[i] >= RASTER_SIZE:
                arr[i] = RASTER_SIZE-1
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
    nadir_result = mask_im.transform((RASTER_SIZE, RASTER_SIZE), ImageTransform.QuadTransform(transform))
    nadir_result_arr = np.array(nadir_result) / 255
    nadir_result_sharp = Image.fromarray((nadir_result_arr > 0.5).astype(np.uint8)*255)
            
    nadir_result.save(f'{result_dir}/nadir_result.png')
    nadir_result_sharp.save(f'{result_dir}/nadir_result_sharp.png')
    return nadir_result_arr