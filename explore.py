


import itertools
import os

import numpy as np
from libs.sosi import read_sos
from utils import get_image_bbox
from tqdm import tqdm
from dbfread import DBF
from shapely.geometry import LineString, MultiLineString

from PIL import Image

Image.MAX_IMAGE_PIXELS = 287944704


im = Image.open('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/data_raw/30243_001_00905_220415_Cam4B.tif')
print(im.size)
im.seek(1)
print(im.size)
exit()


# flight_strips = read_sos('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Flystripe.sos')
# lines = []
# for i in range(1,7):
#     p,q = flight_strips[f'KURVE_{i}']['NØH']
#     lines.append((p[::-1], q[::-1]))

# print(MultiLineString(lines))    
# exit()
# for strip in flight_strips:
#     print(strip)

# files = os.listdir('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/data')
# file_names = [f.split('.')[0] for f in files]

image_data = DBF('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Vertikal-Skrå.dbf')
# image_data = DBF('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Vertikalbildedekning.dbf')

# print(len(image_data))
for i, record in enumerate(image_data):
    print(record)
    print('#####################')
    if i == 5: exit()

# POLYGON ((8.525391 60.841522, 8.525391 61.298749, 10.50293 61.298749, 10.50293 60.841522, 8.525391 60.841522))




# seamlines = list(itertools.chain(*[list(read_sos(path).values())[1:-1] for path in seamline_paths]))
# print(len(seamlines), len(seamlines[0]))
# image_data = np.array(seamlines[::2])
# image_extents = np.array(seamlines[1::2])
# print(image_data.shape, image_extents.shape)
# indices = [i for i, data in enumerate(image_data) if data['ImageName'] in file_names]
# print(indices)
# print(len(indices))
# print('creating polygons')
# image_bboxes = [get_image_bbox(extent) for extent in itertools.chain(image_extents)]

