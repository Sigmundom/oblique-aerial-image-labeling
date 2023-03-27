import io
from imageio.v3 import imread, imwrite
import numpy as np
from rasterio import DatasetReader
import rasterio
import requests
from shapely.geometry import Polygon, box
from geotiff import GeoTiff

url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25832"
coverage =  'dom_25832'
srid = 25832

# minx = 475536.022575551
# miny = 6465689.819260670
# maxx = minx + 100
# maxy = miny + 100


tile_size = 512

img_format = "GeoTiff"
def get_laser_data(area: Polygon) -> DatasetReader:
    return rasterio.open('aoi.tiff')
    # params = {
    #     'service': 'wcs',
    #     'version': '1.0.0',
    #     'request': 'getCoverage',
    #     'coverage': coverage,
    #     'format': img_format,
    #     'width': tile_size,
    #     'height': tile_size,
    #     'crs': f'EPSG:{srid}',
    #     'bbox': ', '.join((str(x) for x in area.bounds))
    # }
    # response = requests.get(url, params=params, stream=True,
    #                        headers=None, timeout=None)
    # if response.status_code == 200:
    #     with open('aoi.tiff', 'wb') as f:
    #         f.write(response.content)
    #         print('File saved successfully.')
    # else:
    #     print(f'Request failed with status code {response.status_code}')
    # image = imread(response.content, extension='.tiff')
    # imwrite('aoi.tiff', image, extension='.tiff')
    # return GeoTiff(io.BytesIO(response.content), as_crs=srid)


if __name__ == '__main__':
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)
    # print()
    # get_laser_data(aoi)
    with rasterio.open('aoi.tiff') as dataset:

        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()
        l, b, r, t = dataset.bounds
        print(l, b, r, t)
        print(dataset.bounds)
        print(dataset.transform)
        print(dataset.crs)

        # # Extract feature shapes and values from the array.
        # for geom, val in features.shapes(
        #         mask, transform=dataset.transform):

        #     # Transform shapes from the dataset's own coordinate
        #     # reference system to CRS84 (EPSG:4326).
        #     geom = rasterio.warp.transform_geom(
        #         dataset.crs, 'EPSG:4326', geom, precision=6)

        #     # Print GeoJSON shapes to stdout.
        #     print(geom)
        # laser_data = GeoTiff('aoi.tiff', as_crs=25832)
        # print(np.array(laser_data.read()))