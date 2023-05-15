import io
import os
from matplotlib import pyplot as plt
import numpy as np
from rasterio import DatasetReader
import rasterio
import requests
from shapely.geometry import Polygon, box
from PIL import Image, ImageFilter

url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25832"
coverage = 'nhm_dom_topo_25832'
srid = 25832




img_format = "GeoTiff"
    # if os.path.exists(file_name):
    #     return rasterio.open(file_name)
def get_heights_tiff(area: Polygon, folder_path=None, tile_size=512) -> DatasetReader:
    # file_name = f'cache/laser_data/heights_{"_".join((format(x, ".2f") for x in area.centroid.coords[0]))}.tiff'

    bounds = area.bounds
    params = {
        'service': 'wcs',
        'version': '1.0.0',
        'request': 'getCoverage',
        'coverage': coverage,
        'format': img_format,
        'width': tile_size,
        'height': tile_size,
        'crs': f'EPSG:{srid}',
        'bbox': ', '.join((format(x, ".2f") for x in bounds))
    }
    response = requests.get(url, params=params, stream=True,
                        headers=None, timeout=None)
    if response.status_code == 200:
        if folder_path is not None:
            with open(os.path.join(folder_path, 'heights.tiff'), 'wb') as f:
                f.write(response.content)
                print('File saved successfully.')
        return rasterio.open(io.BytesIO(response.content))
    else:
        raise requests.HTTPError(f'Request failed with status code {response.status_code}')


if __name__ == '__main__':
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    heights_tiff = get_heights_tiff(aoi)
    heights = heights_tiff.read(1)
    h = heights / heights.max()
    h = h * 255
    h = h.astype(np.uint8)
    im = Image.fromarray(h)
    im_smooth = im.filter(ImageFilter.SMOOTH)
    im_more_smooth = im.filter(ImageFilter.SMOOTH_MORE)

    im_smooth.save('heights_smooth.png')
    im_more_smooth.save('heights_smoothest.png')
    # plt.imsave('heights_relief.png', h)

    # with rasterio.open('aoi.tiff') as dataset:

    #     # Read the dataset's valid data mask as a ndarray.
    #     mask = dataset.dataset_mask()
    #     l, b, r, t = dataset.bounds
    #     print(l, b, r, t)
    #     print(dataset.bounds)
    #     print(dataset.transform)
    #     print(dataset.crs)
