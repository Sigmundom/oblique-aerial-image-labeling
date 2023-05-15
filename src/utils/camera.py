import numpy as np
from math import cos, sin

class Camera():
    def __init__(self, camera_info):
        self.cam_id: str = camera_info['cam_id']
        self.f: float = camera_info['f']
        self.PPx: float = camera_info['PPx']
        self.PPy: float = camera_info['PPy']
        self.width_px: int = camera_info['width_px']
        self.height_px: int = camera_info['height_px']
        self.width_mm: float = camera_info['width_mm']
        self.height_mm: float = camera_info['height_mm']
        self.scale_x: float = self.width_px / self.width_mm
        self.scale_y: float = self.height_px / self.height_mm


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
        ]).T

        def ic_to_wc(x, y, Z):
            if len(x) != len(y) or len(x) != len(Z):
                raise ValueError('The list of coordinates must be of same length')
            
            diff = np.empty((3, len(x)))
            diff[0,:] = x/self.scale_x - self.PPx
            diff[1,:] = y/self.scale_y - self.PPy
            diff[2,:] = -self.f
            d = m[2]@diff

            X = X_0 + (Z-Z_0)*(m[0]@diff / d)
            Y = Y_0 + (Z-Z_0)*(m[1]@diff / d)
            return X, Y
    
        return ic_to_wc
