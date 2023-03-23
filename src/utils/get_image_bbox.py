from collections import OrderedDict

import numpy as np
from shapely.geometry import Polygon


def get_image_bbox(image_extent: OrderedDict):
    image_bbox = np.array(list(image_extent.values())[0], dtype=np.float64)
    image_bbox[:, [1, 0]] = image_bbox[:, [0, 1]]

    image_bbox /= 100 # Coordinates are given in cm for some reason. Convert to m.
    return Polygon(image_bbox)