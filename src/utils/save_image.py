from PIL import Image
import numpy as np

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def save_image(arr, name, mode='L'):
    if mode == '1':
        im = img_frombytes(arr)
    else:
        im = Image.fromarray(((arr/arr.max())*255).astype(np.uint8), mode)
    im.save(name)