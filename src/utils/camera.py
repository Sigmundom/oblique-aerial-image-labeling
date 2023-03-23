import json
import numpy as np
from math import cos, sin

class Camera():
    def __init__(self, camera_info):
        self.cam_id = camera_info['cam_id']
        self.f = camera_info['f']
        self.PPx = camera_info['PPx']
        self.PPy = camera_info['PPy']
        self.width_px = camera_info['width_px']
        self.height_px = camera_info['height_px']
        self.width_mm = camera_info['width_mm']
        self.height_mm = camera_info['height_mm']
        self.scale_x = self.width_px / self.width_mm
        self.scale_y = self.height_px / self.height_mm


    def get_wc_to_ic_transformer(self, X_0, Y_0, Z_0, omega, phi, kappa):
        '''
        Returns a function which transform world coordinates to image coordinates.

            Parameters:
                X_0, Y_0, Z_0 (float): Camera position (easting, northing, height)[m]
                omega, phi, kappa (float): Camera orientation [radians]

            Returns:
                wc_to_ic_transformer (np.array[n, 3] -> np.array[n, 2])
        '''
        P_0 = np.array([X_0, Y_0, Z_0])
        m=np.array([
            [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
            [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
            [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
        ])

        def wc_to_ic(P):
            diff = P-P_0
            d = m[2]@diff.T
            x = (self.PPx - self.f * (m[0]@diff.T / d)) * self.scale_x
            y = (self.PPy - self.f * (m[1]@diff.T / d)) * self.scale_y
            return np.array([x, y]).T
        
        return wc_to_ic
    
    def get_ic_to_wc_transformer(self, X_0, Y_0, Z_0, omega, phi, kappa):
        '''
        Returns a function which transform image coordinates to world coordinates.

            Parameters:
                P_0 (np.array[float,float,float]): Camera position (easting, northing, height)[m]
                omega, phi, kappa (float): Camera orientation [radians]

            Returns:
                wc_to_ic_transformer (np.array[n, 2] -> np.array[n, 3])
        '''
        m=np.array([
            [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
            [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
            [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
        ])

        def ic_to_wc(x, y, Z):
            diff = np.array([(x/self.scale_x - self.PPx), (y/self.scale_y-self.PPy), self.f])
            d = m[2]@diff.T
            X = X_0 + (Z-Z_0)*(m[0]@diff / d)
            Y = Y_0 + (Z-Z_0)*(m[1]@diff / d)
            # def X(Z): 
            #     return X_0 + (Z-Z_0)*(m[0]@diff / d)
            # def Y(Z): 
            #     return Y_0 + (Z-Z_0)*(m[1]@diff / d)
            return X, Y
    
        return ic_to_wc

# def get_wc_to_ic_transformer(image_data):
#     im_height = int(image_data['ImageRows'])
#     im_width = int(image_data['ImageCols'])
#     X_C = float(image_data['CameraX'])
#     Y_C = float(image_data['CameraY'])
#     Z_C = float(image_data['Alt'])
#     P_C = np.array([X_C, Y_C, Z_C])
#     f = float(image_data['FocalLen'])
#     FPx = float(image_data['FPx'])
#     FPy = float(image_data['FPy'])
#     x_0 = float(image_data['PPx'])
#     y_0 = -float(image_data['PPy'])
#     omega = float(image_data['Omega'])
#     phi = float(image_data['Phi'])
#     kappa = float(image_data['Kappa'])

#     m=np.array([
#         [cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)], 
#         [-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
#         [sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]
#     ])

#     def wc_to_ic(P):
#         diff = P-P_C
#         d = m[2]@diff.T
#         x = (x_0 - (f/FPx) * (m[0]@diff.T / d))*im_width
#         y = (y_0 - (f/FPy) * (m[1]@diff.T / d))*im_height
#         return np.array([x, y]).T
        
#     return wc_to_ic


if __name__ == '__main__':
    with open('test_output/image_data/30196_127_02033_210427_Cam4B.json', 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    
    wc_to_ic = get_wc_to_ic_transformer(image_data)
    # wc_to_ic_v2 = get_wc_to_ic_transformer_v2(image_data)

    print(wc_to_ic(np.array([475843.91, 6465987.86, 10])))
    # print(wc_to_ic_v2(np.array([475843.91, 6465987.86, 10])))