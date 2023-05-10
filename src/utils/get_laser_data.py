import io
import os
from rasterio import DatasetReader
import rasterio
import requests
from shapely.geometry import Polygon, box

url = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dom-nhm-25832"
# coverage =  'dom_25832'
coverage = 'nhm_dom_topo_25832'
srid = 25832



tile_size = 512

img_format = "GeoTiff"
def get_laser_data(area: Polygon) -> DatasetReader:
    file_name = f'cache/laser_data/heights_{"_".join((format(x, ".2f") for x in area.centroid.coords[0]))}.tiff'
    # if os.path.exists(file_name):
    #     return rasterio.open(file_name)
    if False:
        pass
    else:
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
            with open(file_name, 'wb') as f:
                f.write(response.content)
                print('File saved successfully.')
            return rasterio.open(io.BytesIO(response.content))
        else:
            raise requests.HTTPError(f'Request failed with status code {response.status_code}')
        # image = imread(response.content, extension='.tiff')
        # imwrite('aoi.tiff', image, extension='.tiff')


if __name__ == '__main__':
    x = 476155.42
    y = 6465926.15
    d = 25
    aoi = box(x-d, y-d, x+d, y+d)

    with rasterio.open('aoi.tiff') as dataset:

        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()
        l, b, r, t = dataset.bounds
        print(l, b, r, t)
        print(dataset.bounds)
        print(dataset.transform)
        print(dataset.crs)
