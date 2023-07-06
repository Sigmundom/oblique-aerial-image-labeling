import io
import os
from matplotlib import pyplot as plt
import numpy as np
from rasterio import DatasetReader
import rasterio
import requests
from shapely.geometry import Polygon, box
# from PIL import Image, ImageFilter

from utils import ensure_folder_exists

url = "https://wms.geonorge.no/skwms1/wms.nib"

def get_orthophoto(area: Polygon, height: int, width: int) -> DatasetReader:
    file_name = f'cache/ortofoto/{"{:.0f}_{:.0f}".format(*area.centroid.coords[0])}.jpeg'
    if os.path.exists(file_name):
        return file_name

    bounds = area.bounds
    params = {
        'request': 'getMap',
        'format': "image/jpeg",
        'width': width,
        'height': height,
        'crs': 'EPSG:25832',
        'layers': 'ortofoto',
        'bbox': ', '.join((format(x, ".2f") for x in bounds))
    }
    response = requests.get(url, params=params, stream=True,
                        headers=None, timeout=None)
    if response.status_code == 200:
        ensure_folder_exists(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return io.BytesIO(response.content)
    else:
        raise requests.HTTPError(f'Request failed with status code {response.status_code}')


if __name__ == '__main__':
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    im = get_orthophoto(aoi)
    im.save('ortofoto.jpeg')
    exit()
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
