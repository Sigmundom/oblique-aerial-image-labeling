import json
import numpy as np

area = [
    [476034.3338150481, 6466232.680869672],
    [475740.02203449875, 6465793.010778319],
    [476020.66183297796, 6465640.759635763],
    [476315.19476801465, 6466120.643464423],
    [476034.3338150481, 6466232.680869672],
]
x_coords = [c[0] for c in area]
y_coords = [c[1] for c in area]
x_min = min(x_coords)
x_max = max(x_coords)
y_min = min(y_coords)
y_max = max(y_coords)

with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json') as f:
    city_json = json.load(f)


v_all = city_json['vertices']

v_in_image = np.array([xy for xy in v_all if xy[0] > x_min and xy[0] < x_max and xy[1] > y_min and xy[1] < y_max])

print(v_in_image[:,2].mean()) 
np.save('vertices', v_in_image)
