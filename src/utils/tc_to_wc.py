import json
from math import cos, sin
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from . import get_wc_to_ic_transformer

image_folder = 'test_output/images'
image_data_folder = 'test_output/image_data'

def get_ic_to_wc_transformer(image_data):
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
    x_0 = PPx
    y_0 = PPy 
    omega = -float(image_data['Omega'])
    phi = -float(image_data['Phi'])
    kappa = -float(image_data['Kappa'])

    m=np.array([
        [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
        [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
        [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
    ])

    def ic_to_wc(x, y):
        diff = np.array([(x/im_width-x_0)*FPx, (y/im_height-y_0)*FPy, f])
        d = m[2]@diff
        def X(Z): 
            return X_C + (Z-Z_C)*(m[0]@diff / d)
        def Y(Z): 
            return Y_C + (Z-Z_C)*(m[1]@diff / d)
        return X, Y
    
    return ic_to_wc

def tile_coordinate_to_wc(tile_xy, tile_name):
    name_parts = tile_name.split('.')[0].split('_')
    image_name = '_'.join(name_parts[:-1])
    tile_index = int(name_parts[-1])
    
    with open(f'{image_data_folder}/{image_name}.json', 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    
    tile_data = image_data['tiles']
    tile_h, tile_w = tile_data['size']
    anchor = tile_data['anchors'][tile_index]
    
    x_ic, y_ic = tile_xy[0]+anchor[0], anchor[1]-tile_xy[1]
    print(x_ic, y_ic)
    ic_to_wc = get_ic_to_wc_transformer(image_data)
    X_z, Y_z = ic_to_wc(x_ic, y_ic)
    print('Z=10:', X_z(10), Y_z(10))

    wc_to_ic = get_wc_to_ic_transformer(image_data)
    Z = np.linspace(0,20, num=10)
    X_wc = X_z(Z)
    Y_wc = Y_z(Z)
    print('WC: ', X_wc, Y_wc)
    # x, y = wc_to_ic(np.array([X_z(10), Y_z(10), 10]))
    image_xy = np.array([wc_to_ic(np.array([x,y,z])) for x, y, z in zip(X_wc, Y_wc,Z)])
    print('IC:', image_xy)

    tile_x = image_xy[:,0] - anchor[0]
    tile_y = anchor[1] - image_xy[:,1]
    # x -= anchor[1]
    # y -= anchor[0]
    print('TC:', tile_x, tile_y)
    # print(x)
    # print(y)
    # print(x,y)
    # exit()
    # print(wc_to_ic(np.array([475843.91, 6465987.86, 10])))
    # print(XX)
    # print(wc_to_ic(YY))
    # exit()
    plt.imshow(tile, extent=[0, tile_h, tile_w, 0])
    plt.scatter(*point, c='r')
    plt.scatter(tile_x, tile_y, c='purple')
    plt.show()

    return X_wc, Y_wc



    




if __name__ == '__main__':
    tile_name = '30196_127_02033_210427_Cam4B_0.jpg'
    point = (100, 159) #x, y
    tile = Image.open(f'{image_folder}/{tile_name}')
    X, Y = tile_coordinate_to_wc(point, tile_name)

    