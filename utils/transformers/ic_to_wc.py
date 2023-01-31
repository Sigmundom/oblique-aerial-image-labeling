import json
import numpy as np
from math import cos, sin

def get_ic_to_wc_transformer(image_data):
    im_height = int(image_data['ImageRows'])
    im_width = int(image_data['ImageCols'])
    X_C = float(image_data['CameraX'])
    Y_C = float(image_data['CameraY'])
    Z_C = float(image_data['Alt'])
    P_C = np.array([X_C, Y_C, Z_C])
    f = float(image_data['FocalLen'])
    FPx = float(image_data['FPx'])
    FPy = float(image_data['FPy'])
    x_0 = float(image_data['PPx'])
    y_0 = -float(image_data['PPy'])
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
        x = (x_0 - (f/FPx) * (m[0]@diff.T / d))*im_width
        y = (y_0 - (f/FPy) * (m[1]@diff.T / d))*im_height
        return np.array([x, y]).T
        
    return wc_to_ic


if __name__ == '__main__':
    with open('test_output/image_data/30196_127_02033_210427_Cam4B.json', 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    
    ic_to_wc = get_ic_to_wc_transformer(image_data)

    print(wc_to_ic(np.array([475843.91, 6465987.86, 10])))
    # print(wc_to_ic_v2(np.array([475843.91, 6465987.86, 10])))