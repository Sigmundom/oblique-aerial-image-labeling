import itertools

import numpy as np
from libs.sosi import read_sos
from shapely.strtree import STRtree
from shapely.geometry import Polygon

from utils import get_image_bbox


class ImageIndex():
    def __init__(self, seamline_paths: list[str]):
        seamlines = [list(read_sos(path).values())[1:-1] for path in seamline_paths]
        self.image_data = np.array(itertools.chain([s[::2] for s in seamlines]))
        image_extents = [s[1::2] for s in seamlines]
        print('creating polygons')
        image_bboxes = [get_image_bbox(extent) for extent in itertools.chain(image_extents)]
        print('creating tree')
        self.tree = STRtree(image_bboxes)

    def get_images_covering_polygon(self, polygon: Polygon):
        indexes = self.tree.query(polygon)
        return self.image_data.take(indexes)






if __name__ == '__main__':
    seamline_paths = [
        'data/Somlinjefiler/cam4B.sos',
        'data/Somlinjefiler/cam5R.sos',
        'data/Somlinjefiler/cam6L.sos',
        'data/Somlinjefiler/cam7F.sos'
    ]

    index = ImageIndex(seamline_paths)
    index.get_images_covering_polygon()