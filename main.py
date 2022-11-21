import datetime
from matplotlib.patches import Polygon, Rectangle
from create_annotation import create_annotation
import sosi
import json
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from os import path, makedirs
import config
from split_image import split_image, get_tile
from PIL import Image


def get_image_data(image_name):
    # Read and find data

    data_file = f'{image_name}.json'
    
    if path.exists(data_file):
        # Load data
        with open(data_file, encoding='utf8') as f:
            return json.load(f)


    cam = image_name.split('_')[-1]
    sos_file = f'images/Somlinjefiler/{cam.replace(cam[0], cam[0].lower(), 1)}.sos'
    data = sosi.read_sos(sos_file)

    data_values = data.values()
    image_data = next((x for x in data_values if 'ImageName' in x and x['ImageName'] == image_name), None)

    return image_data

def get_v_in_image(image_data, image_file):
    v_file = image_file.replace('.jpg', '.npy')
    
    if path.exists(v_file):
        return np.load(v_file)

    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        city_json = json.load(f)

    ul = [float(image_data["ULx"]), float(image_data["ULy"])]
    ur = [float(image_data["URx"]), float(image_data["URy"])]
    lr = [float(image_data["LRx"]), float(image_data["LRy"])]
    ll = [float(image_data["LLx"]), float(image_data["LLy"])]
    x_coords = [x[0] for x in [ul, ur, lr, ll]]
    y_coords = [x[1] for x in [ul, ur, lr, ll]]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    v_all = city_json['vertices']

    v_in_image = np.array([xy for xy in v_all if xy[0] > x_min and xy[0] < x_max and xy[1] > y_min and xy[1] < y_max])
    
    np.save(v_file, v_in_image)

    return v_in_image


def create_output_folder():
    folders = [
        config.output_folder, 
        f'{config.output_folder}/images',
        f'{config.output_folder}/images/train',
        f'{config.output_folder}/annotations',
        f'{config.output_folder}/image_data',
        ]
    for folder in folders:
        if not path.exists(folder):
            makedirs(folder)




def get_buildings_in_image(image_data, city_json, wc_to_ic):
    corner_points = ["UL", "UR", "LR", "LL"]
    x_coords = [float(image_data[f'{point}x']) for point in corner_points]
    y_coords = [float(image_data[f'{point}y']) for point in corner_points]
    x, x_max = min(x_coords), max(x_coords)
    y, y_max = min(y_coords), max(y_coords)

    all_buildings = city_json['CityObjects']
    all_vertices = np.array(city_json['vertices'])
    vertices_in_image = (all_vertices[:, 0] > x) & (all_vertices[:,0] < x_max) & (all_vertices[:,1] > y) & (all_vertices[:, 1] < y_max)
    buildings_in_image = []

    for building in all_buildings.values():
        geometry = building["geometry"]
        if len(geometry) > 1:
            print("Length is:", len(geometry))
        if len(geometry) == 0:
            continue
        boundaries = geometry[0]['boundaries']
        vertices_i = [v for boundary in boundaries for v in boundary[0]]
        
        if np.any(np.take(vertices_in_image, vertices_i, 0)):
            surfaces = []
            for boundary in boundaries:
                surfaces.append([wc_to_ic(*all_vertices[v_i]) for v_i in boundary[0]])
            vertices = np.array([v for surface in surfaces for v in surface])
            x = vertices[:,0].min()
            x_max = vertices[:,0].max()
            y = vertices[:,1].min()
            y_max = vertices[:,1].max()
            building['surfaces'] = surfaces
            building['bbox'] = (x, y, x_max-x, y_max-y)
            buildings_in_image.append(building)
    
    return buildings_in_image


def get_buildings_in_tile(buildings, anchor):
    ay, ax = anchor
    a_height, a_width = config.tile_size
    def is_building_in_tile(building):
        bx, by, b_width, b_height = building['bbox']
        return not (
            bx + b_width < ax or
            by + b_height < ay or 
            bx > ax + a_width or 
            by > ay + a_height
            )

    return list(filter(is_building_in_tile, buildings))


    

def get_wc_to_ic_transformer(image_data):
    im_height = int(image_data['ImageRows'])
    im_width = int(image_data['ImageCols'])
    X_C = float(image_data['CameraX'])
    Y_C = float(image_data['CameraY'])
    Z_C = float(image_data['Alt'])
    P_C = np.array([X_C, Y_C, Z_C])
    f = float(image_data['FocalLen'])
    FPx = float(image_data['FPx'])
    FPy = float(image_data['FPy'])
    PPx = float(image_data['PPx'])
    PPy = -float(image_data['PPy'])
    x_0 = PPx*FPx
    y_0 = PPy*FPy 
    omega = float(image_data['Omega'])
    phi = float(image_data['Phi'])
    kappa = float(image_data['Kappa'])

    m=np.array([
        [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
        [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
        [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
    ])

    def wc_to_ic(P):
        diff = P-P_C
        d = m[2]@diff.T
        x = (x_0 - f * (m[0]@diff.T / d))*im_width/FPx
        y = (y_0 - f * (m[1]@diff.T / d))*im_height/FPy
        return x, y
    
    return wc_to_ic


    


def process_image(image_file, city_json):
    # image = plt.imread(image_file)
    image = Image.open(image_file)
    image_name = path.basename(image_file).split('.')[0]

    image_data = get_image_data(image_name)
    date, time = image_data['ShotDate']
    year, month, day = [int(x) for x in date.split('/')]
    h = int(time[:2])
    m = int(time[2:4])
    s = int(time[4:6])
    date_captured = datetime.datetime(year, month, day, h, m, s)

    wc_to_ic = get_wc_to_ic_transformer(image_data)

    
    buildings_in_image = get_buildings_in_image(image_data, city_json, wc_to_ic)
    
    anchors = split_image(image)
    tile_names = []
    annotations = []
    images = []

    image_folder = f'{config.output_folder}/images/train'
    
    annotation_id = 1
    image_id = 1

    first = True
    for anchor in anchors:
        tile_name = f'{image_name}_{anchor[0]}_{anchor[1]}'
        tile_names.append(tile_name)
        buildings_in_tile = get_buildings_in_tile(buildings_in_image, anchor)
        if len(buildings_in_tile) == 0:
            continue
        if first:
            first = False
            continue
        tile = get_tile(image, anchor)

        all_surfaces = []
        for building in buildings_in_tile:
            surfaces = [[[x-anchor[1], y-anchor[0]] for x,y  in surface] for surface in building['surfaces']]
            all_surfaces.append(surfaces)
            annotation = create_annotation(surfaces, image_id, annotation_id)
            if annotation is not None:
                annotations.append(annotation)
                annotation_id += 1
        
        tile_file = tile_name + '.jpg'
        tile.save(f'{image_folder}/{tile_file}')
        images.append({
            "id": image_id,
            "height": config.tile_size[0],
            "width": config.tile_size[1],
            "file_name": tile_file,
            "license": 1,
            "date_captured": str(date_captured)
        })
        image_id += 1


    coco = {}
    coco["info"] = {}
    coco['images'] = images
    coco['annotations'] = annotations
    coco['licences'] = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        }
    ]
    coco['categories'] = [
        {
            "id": 0,
            "name": "Builing",
        }
    ]
    


    with open(f'{config.output_folder}/annotations/instances_train.json', 'w', encoding='utf8') as f:
        json.dump(coco, f)
    
    image_data['tiles'] = tile_names
    
    # Save metadata
    with open(f'{config.output_folder}/image_data/{image_name}.json', 'w', encoding='utf8') as f:
        json.dump(image_data, f)
    


def main():
    image_file = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    # image_file = 'images/Framoverrettede bilder/30196_127_02023_210427_Cam7F.jpg'
    # image_file = 'images/Framoverrettede bilder/30196_127_02026_210427_Cam7F.jpg'
    # image_file = 'images/Høyrerettede bilder/30196_124_01897_210427_Cam5R.jpg'
    # image_file = 'images/Høyrerettede bilder/30196_124_01898_210427_Cam5R.jpg'
    # image_file = 'images/Venstrerettede bilder/30196_128_02062_210427_Cam6L.jpg'
    # image_file = 'images/Venstrerettede bilder/30196_130_02151_210427_Cam6L.jpg'
    # image_file= 'images/Venstrerettede bilder/30196_130_02153_210427_Cam6L.jpg'
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        city_json = json.load(f)
    create_output_folder()
    process_image(image_file, city_json)
    
    exit()
    image_name = path.basename(image_file).split('.')[0]
    image_data = get_image_data(image_name)

    # building_vertices = np.load('vertices.npy')
    building_vertices = get_v_in_image(image_data, image_file)

    image = plt.imread(image_file)

    im_height = int(image_data['ImageRows'])
    im_width = int(image_data['ImageCols'])
    assert image.shape[:2] == (im_height, im_width), 'Image shape does not match given shape in the .sos file'


    X_C = float(image_data['CameraX'])
    Y_C = float(image_data['CameraY'])
    Z_C = float(image_data['Alt'])
    P_C = np.array([X_C, Y_C, Z_C])
    f = float(image_data['FocalLen'])
    FPx = float(image_data['FPx'])
    FPy = float(image_data['FPy'])
    PPx = float(image_data['PPx'])
    PPy = -float(image_data['PPy'])
    x_0 = PPx*FPx
    y_0 = PPy*FPy 
    omega = float(image_data['Omega'])
    phi = float(image_data['Phi'])
    kappa = float(image_data['Kappa'])

    # m=np.array([
    #     [cos(phi)*cos(kappa), -cos(phi)*sin(kappa), sin(phi)], 
    #     [cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), -sin(omega)*cos(phi)],
    #     [sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa), cos(omega)*cos(phi)]
    # ])
    m=np.array([
        [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
        [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
        [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
    ])

    def wc_to_ic(P):
        diff = P-P_C
        d = m[2]@diff.T
        x = (x_0 - f * (m[0]@diff.T / d))*im_width/FPx
        y = (y_0 - f * (m[1]@diff.T / d))*im_height/FPy
        return x, y



    points = np.array([wc_to_ic(v) for v in building_vertices])
    print(np.histogram(points))

    points = points[(-im_width/2 < points[:,0]) & (points[:,0] < im_width/2) & (-im_height/2 < points[:,1]) & (points[:,1] < im_height/2)]

    x, y = points.T

    plt.imshow(image, extent=[-im_width/2., im_width/2., -im_height/2., im_height/2. ])
    plt.scatter(x,y, marker='.', color="red", s=0.8)
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()



# def create_mask(image, wc_to_ic, buildings, city_json):
#     mask = np.zeros_like(image)
#     all_vertices = np.array(city_json['vertices'])
#     fig, ax = plt.subplots()
#     ax.imshow(image, extent=[-mask.shape[1]/2., mask.shape[1]/2., -mask.shape[0]/2., mask.shape[0]/2. ])
#     for building in buildings.values():
#         geometry = building['geometry'][0]
#         boundaries = geometry['boundaries']
#         surfaces = geometry['semantics']['surfaces']
#         walls = []
#         roofs = []
#         for boundary, surface in zip(boundaries, surfaces):
#             vertices = [wc_to_ic(*all_vertices[v_i]) for v_i in boundary[0]]
#             if surface['type'] == 'RoofSurface':
#                 roofs.append(vertices)
#             elif surface['type'] == 'WallSurface':
#                 walls.append(vertices)
#             else:
#                 print('New surface type!', surface['type'])

#         for wall in walls:
#             ax.add_patch(Polygon(wall, closed=True, facecolor='red'))
#         for roof in roofs:
#             ax.add_patch(Polygon(roof, closed=True, facecolor='blue'))

            

    # colors = 100 * np.random.rand(len(patches))
    # p.set_array(colors)
    # fig, ax = plt.subplots(7700, 10300)
    # fig.colorbar(p, ax=ax)
    # p = PatchCollection(patches)
    # fig.canvas.draw()
    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # w, h = fig.canvas.get_width_height()
    # im = data.reshape((int(h), int(w), -1))
    # plt.imsave('image.png', im)
    # plt.show()


if __name__ == '__main__':
    main()















