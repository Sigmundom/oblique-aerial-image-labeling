from typing import TypedDict
from libs.sosi import read_sos
from dbfread import DBF
from math import radians


class ImageDataRecord(TypedDict):
    image_id: str
    x: float
    y: float
    height: float 
    omega: float #radians
    phi: float #radians
    kappa: float #radians
    cam: str


class ImageData():
    def __init__(self, records):
        self._records: dict[str, ImageDataRecord] = records

    def __getitem__(self, image_id: str):
        return self._records.get(image_id)

    def __len__(self):
        return len(self._records)

    @classmethod
    def from_sos(cls, *paths):
        records = {}
        for path in paths:
            seamline = list(read_sos(path).values())[1:-1]
            data = seamline[::2]
            for e in data:
                records[e['ImageName']] = ImageDataRecord(
                    x=float(e['CameraX']),
                    y=float(e['CameraY']),
                    height=float(e['Alt']),
                    omega=float(e['Omega']),
                    phi=float(e['Phi']),
                    kappa=float(e['Kappa']),
                    cam=float()
                )
        return cls(records)
    
    @classmethod
    def from_dbf(cls, *paths):
        records = {}
        for path in paths:
            data = DBF(path)
            for e in data:
                records[e['imageid']] = ImageDataRecord(
                    x=e['easting'],
                    y=e['northing'],
                    height=e['height'],
                    omega=radians(e['omega']),
                    phi=radians(e['phi']),
                    kappa=radians(e['kappa']),
                )
            
        return cls(records)

if __name__ == '__main__':
    # data = ImageData.from_sos('data/Somlinjefiler/cam4B.sos', 'data/Somlinjefiler/cam5R.sos')
    data = ImageData.from_dbf('/media/sigmundmestad/aa7e1253-187b-48d4-af32-eb9f050db5dd/sigmunom/Dekningsoversikt/TT-30243_Vertikal-Skrå.dbf')
    print(data['30243_001_00905_220415_Cam0N'])