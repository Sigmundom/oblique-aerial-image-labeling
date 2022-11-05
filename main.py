from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import sosi
import json
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from os import path, makedirs
from PIL import Image
import config
from split_image import split_image


OUTPUT_FOLDER = 'output'


def get_image_data(image_name):
    # Read and find data

    data_file = f'{image_name}.json'
    
    if path.exists(data_file):
        # Load data
        with open(data_file) as f:
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

    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json') as f:
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
        f'{config.output_folder}/image_data',
        f'{config.output_folder}/masks',
        ]
    for folder in folders:
        if not path.exists(folder):
            makedirs(folder)

# def get_buildings_in_bbox(buildings, all_vertices, x_min, x_max, y_min, y_max):




def get_buildings_in_image(all_buildings, all_vertices, image_data):
    corner_points = ["UL", "UR", "LR", "LL"]
    x_coords = [float(image_data[f'{point}x']) for point in corner_points]
    y_coords = [float(image_data[f'{point}y']) for point in corner_points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    vertices_in_image = (all_vertices[:, 0] > x_min) & (all_vertices[:,0] < x_max) & (all_vertices[:,1] > y_min) & (all_vertices[:, 1] < y_max)
    buildings_in_image = {}

    for id, building in all_buildings.items():
        geometry = building["geometry"]
        if len(geometry) > 1:
            print("Length is:", len(geometry))
        if len(geometry) == 0:
            continue
        boundaries = geometry[0]['boundaries']
        vertices_i = [v for boundary in boundaries for v in boundary[0]]
        
        if np.any(np.take(vertices_in_image, vertices_i, 0)):
            buildings_in_image[id]=building
    
    return buildings_in_image

    # return get_buildings_in_bbox(all_buildings, all_vertices, x_min, x_max, y_min, y_max)


    

def get_wc_to_ic_transformer(image_data):
    im_height = int(image_data['ImageRows'])
    im_width = int(image_data['ImageCols'])
    X_C = float(image_data['CameraX'])
    Y_C = float(image_data['CameraY'])
    Z_C = float(image_data['Alt'])
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

    def wc_to_ic(X,Y,Z):
        dX = X-X_C
        dY = Y-Y_C
        dZ = Z-Z_C
        diff = np.array([dX, dY, dZ])
        d = m[2]@diff.T
        x = (x_0 - f * (m[0]@diff.T / d))*im_width/FPx
        y = (y_0 - f * (m[1]@diff.T / d))*im_height/FPy
        return x, y
    
    return wc_to_ic

def create_mask(image, wc_to_ic, buildings, city_json):
    mask = np.zeros_like(image)
    all_vertices = np.array(city_json['vertices'])
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[-mask.shape[1]/2., mask.shape[1]/2., -mask.shape[0]/2., mask.shape[0]/2. ])
    for building in buildings.values():
        geometry = building['geometry'][0]
        boundaries = geometry['boundaries']
        surfaces = geometry['semantics']['surfaces']
        walls = []
        roofs = []
        for boandary, surface in zip(boundaries, surfaces):
            vertices = [wc_to_ic(*all_vertices[v_i]) for v_i in boandary[0]]
            if surface['type'] == 'RoofSurface':
                roofs.append(vertices)
            elif surface['type'] == 'WallSurface':
                walls.append(vertices)
            else:
                print('New surface type!', surface['type'])

        for wall in walls:
            ax.add_patch(Polygon(wall, closed=True, facecolor='red'))
        for roof in roofs:
            ax.add_patch(Polygon(roof, closed=True, facecolor='blue'))

            

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
    plt.show()
    


def process_image(image_file, city_json):
    image = plt.imread(image_file)
    image_name = path.basename(image_file).split('.')[0]

    image_data = get_image_data(image_name)
    wc_to_ic = get_wc_to_ic_transformer(image_data)

    
    all_buildings = city_json['CityObjects']
    all_vertices = np.array(city_json['vertices'])
    buildings_in_image = get_buildings_in_image(all_buildings, all_vertices, image_data)
    create_mask(image, wc_to_ic, buildings_in_image, city_json)
    
    anchors = split_image(image)

    tile_names = []
    for anchor in anchors:
        tile_name = f'{image_name}_{anchor[0]}_{anchor[1]}'
        tile_names.append(tile_name)
        tile = image[anchor[0]:anchor[0]+config.tile_size[0], anchor[1]:anchor[1]+config.tile_size[1]]
        # buildings_in_tile = get_buildings_in_bbox(buildings_in_image, all_vertices,)
        # mask = create_mask(tile, image_data, buildings, city_json)
        
        
        plt.imsave(f'{config.output_folder}/images/{tile_name}.jpg', tile)

    image_data['tiles'] = tile_names

    # Save metadata
    with open(f'{config.output_folder}/image_data/{image_name}.json', 'w') as f:
        json.dump(image_data, f)
    


def main():
    # image_file = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    # image_file = 'images/Framoverrettede bilder/30196_127_02023_210427_Cam7F.jpg'
    image_file = 'images/Framoverrettede bilder/30196_127_02026_210427_Cam7F.jpg'
    # image_file = 'images/Høyrerettede bilder/30196_124_01897_210427_Cam5R.jpg'
    # image_file = 'images/Høyrerettede bilder/30196_124_01898_210427_Cam5R.jpg'
    # image_file = 'images/Venstrerettede bilder/30196_128_02062_210427_Cam6L.jpg'
    # image_file = 'images/Venstrerettede bilder/30196_130_02151_210427_Cam6L.jpg'
    # image_file= 'images/Venstrerettede bilder/30196_130_02153_210427_Cam6L.jpg'
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json') as f:
        city_json = json.load(f)
    create_output_folder()
    process_image(image_file, city_json)
    exit()

    image_data = get_image_data(image_file)

    # building_vertices = np.load('vertices.npy')
    building_vertices = get_v_in_image(image_data, image_file)
    
    image = plt.imread(image_file)

    im_height = int(image_data['ImageRows'])
    im_width = int(image_data['ImageCols'])
    assert image.shape[:2] == (im_height, im_width), 'Image shape does not match given shape in the .sos file'


    X_C = float(image_data['CameraX'])
    Y_C = float(image_data['CameraY'])
    Z_C = float(image_data['Alt'])
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

    def wc_to_ic(X,Y,Z):
        dX = X-X_C
        dY = Y-Y_C
        dZ = Z-Z_C
        diff = np.array([dX, dY, dZ])
        d = m[2]@diff.T
        x = (x_0 - f * (m[0]@diff.T / d))*im_width/FPx
        y = (y_0 - f * (m[1]@diff.T / d))*im_height/FPy
        return x, y


    points = np.array([wc_to_ic(*v) for v in building_vertices])

    points = points[(-im_width/2 < points[:,0]) & (points[:,0] < im_width/2) & (-im_height/2 < points[:,1]) & (points[:,1] < im_height/2)]

    x, y = points.T

    plt.imshow(image, extent=[-im_width/2., im_width/2., -im_height/2., im_height/2. ])
    plt.scatter(x,y, marker='.', color="red", s=0.8)
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()




if __name__ == '__main__':
    main()















