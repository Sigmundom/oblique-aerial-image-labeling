import json
import os
import urllib
import shapely.geometry as sg

from utils import ensure_folder_exists


Municipality = dict(name=str, age=int)

def get_municipality_border(municipality: Municipality) -> sg.Polygon: 
    number, name = municipality['number'], municipality['name']
    file_path = f'data/municipalities/{number}-{name}.json'
    if os.path.exists(file_path):
        with open(file_path, encoding='utf8') as f:
            geojson = json.load(f)
    else:
        response = urllib.request.urlopen(f'https://ws.geonorge.no/kommuneinfo/v1/kommuner/{number}/omrade?utkoordsys=25832', ).read()
        geojson = json.loads(response)
        ensure_folder_exists('data/municipalities')
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(geojson, f)
    
    return sg.Polygon(geojson['omrade']['coordinates'][0][0])