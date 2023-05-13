import json
import numpy as np

import requests

np.set_printoptions(suppress = True)

url = 'https://ws.geonorge.no/hoydedata/v1/datakilder/dtm1/punkt'

def get_terrain_heights(coordinates):
    params = {
            'datakilde': 'dtm1',
            'koordsys': 25832,
            'punkter': json.dumps(coordinates)
        }
    response = requests.get(url, params=params, headers=None, timeout=None)
    return np.array([[p['x'], p['y'], p['z'] if p['z'] is not None else 0] for p in json.loads(response.content)['punkter']])


if __name__ == '__main__':
    # coordinates = [
    #     [
    #     412950,
    #     6451550
    #     ],
    #     [
    #     412950,
    #     6452350
    #     ],
    #     [
    #     413600,
    #     6452350
    #     ],
    #     [
    #     413600,
    #     6451550
    #     ],
    #     [
    #     412950,
    #     6451550
    #     ]
    # ]
    coordinates = [
        [
        411962,
        6456331
        ],
        [
        411976,
        6456934
        ],
        [
        412657,
        6456919
        ],
        [
        412644,
        6456316
        ],
        [
        411962,
        6456331
        ]
    ]
    print(get_terrain_heights(coordinates))
        