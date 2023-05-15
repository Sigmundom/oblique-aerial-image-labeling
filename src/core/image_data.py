
import json
from libs.sosi import read_sos
from math import radians
import os
import shapefile
import utm
from shapely.geometry import Polygon
from shapely.validation import make_valid

from utils import get_image_bbox, Camera
import re

def get_camera(cameras, image_name) -> Camera:
    cam_id = re.search("cam\d[BFRLN]", image_name, re.IGNORECASE)
    if cam_id is None:
        raise ValueError(f'Image name "{image_name}" does not contain a valid camera_id')
    cam_id = cam_id.group()
    if cam_id in cameras:
        return cameras[cam_id]

class ImageDataRecord():
    def __init__(self, 
                 image_name: str,
                 image_path: str, 
                 x: float, 
                 y:float, 
                 height:float, 
                 omega: float, 
                 phi: float, 
                 kappa: float, 
                 cam: Camera, 
                 bbox: Polygon = None,
                 ):
        self.name = image_name
        self.path: str = image_path
        self.x = x
        self.y = y
        self.height = height
        self.omega = omega
        self.phi = phi
        self.kappa = kappa
        self.cam = cam
        self.wc_to_ic = cam.get_wc_to_ic_transformer(x, y, height, omega, phi, kappa)
        self.bbox: Polygon = bbox

    @property
    def ic_to_wc(self):
        return self.cam.get_ic_to_wc_transformer(self.x, self.y, self.height, self.omega, self.phi, self.kappa)
                        
                 
    def is_image_in_area(self, area: Polygon) -> bool:
        intersection = self.bbox.intersection(area)
        overlap = intersection.area / self.bbox.area
        if overlap > 0.99:
            return True
        elif overlap > 0.5:
            print('Partial overlap:', overlap)
            return False
        
    def save_image_data(self, folder, tile_size):
        with open(f'{folder}/{self.name}.json', 'w', encoding='utf8') as f:
            json.dump({
                'image_name': self.name,
                'x': self.x,
                'y': self.y,
                'height': self.height,
                'omega': self.omega,
                'phi': self.phi,
                'kappa': self.kappa,
                'cam_id': self.cam.cam_id,
                'f': self.cam.f,
                'PPx': self.cam.PPx,
                'PPy': self.cam.PPy,
                'width_px': self.cam.width_px,
                'height_px': self.cam.height_px,
                'width_mm': self.cam.width_mm,
                'height_mm': self.cam.height_mm,
                'tile_size': tile_size
            }, f)


class ImageDataList():
    def __init__(self, data: list[ImageDataRecord]):
        self._data = data

    def __getitem__(self, image_id: str):
        return self._data.get(image_id)

    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    @classmethod
    def from_sos(cls, image_paths, image_data_paths, cameras):
        available_images = set([os.path.basename(path).split('.')[0] for path in image_paths])
        data = []
        for path in image_data_paths:
            seamline = list(read_sos(path).values())[1:-1]
            image_data = seamline[::2]
            image_extents = seamline[1::2]
            for rec, extent in zip(image_data, image_extents):
                image_name = rec['ImageName']
                if image_name in available_images:
                    cam = get_camera(cameras, image_name)
                    if cam is None: continue
                    image_path = next(filter(lambda path: image_name in path, image_paths), None)
                    data.append(ImageDataRecord(
                        image_name=image_name,
                        image_path=image_path,
                        x=float(rec['CameraX']),
                        y=float(rec['CameraY']),
                        height=float(rec['Alt']),
                        omega=float(rec['Omega']),
                        phi=float(rec['Phi']),
                        kappa=float(rec['Kappa']),
                        cam=cam,
                        bbox=get_image_bbox(extent)
                    ))
        return cls(data)


 
    
    @classmethod
    def from_shp(cls, image_paths, image_data_paths, cameras):
        available_images = set([os.path.basename(path).split('.')[0] for path in image_paths])
        data = []
        for path in image_data_paths:
            sf = shapefile.Reader(path)
            
            for rec in sf.iterRecords(fields=['imageid']):
                image_name = rec['imageid']
                if rec['imageid'] in available_images:
                    cam = get_camera(cameras, image_name)
                    if cam is None: continue
                    # if not cam.cam_id == 'cam6L': continue

                    image_path = next(filter(lambda path: image_name in path, image_paths), None)
                    shapeRec = sf.shapeRecord(rec.oid)
                    bbox = Polygon([utm.from_latlon(*point[::-1], force_zone_number=32, force_zone_letter='N')[:2] for point in shapeRec.shape.points])
                    rec = shapeRec.record
                    data.append(ImageDataRecord(
                        image_name=image_name,
                        image_path=image_path,
                        x=rec['easting'],
                        y=rec['northing'],
                        height=rec['height'],
                        omega=radians(rec['omega']),
                        phi=radians(rec['phi']),
                        kappa=radians(rec['kappa']),
                        cam=cam,
                        bbox=bbox
                    ))
            
        return cls(data)

# if __name__ == '__main__':
    # data = ImageData.from_sos('data/Somlinjefiler/cam4B.sos', 'data/Somlinjefiler/cam5R.sos')
    # image_paths = os.listdir('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/data')
    # data_paths = ['/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Vertikal-Skrå.dbf']
    # # data = ImageData.from_dbf(image_paths, data_paths)

    # print(data._records)
    # print(data['30243_001_00905_220415_Cam0N'])