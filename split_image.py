import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math
from config import tile_size, min_overlap

# def split_image(image:np.ndarray, tile_size: tuple[int, int], min_overlap: Union[int, tuple[int,int]] = (0,0)):


def split_image(image:np.ndarray):
    num_tiles = (math.ceil(image.shape[0] / (tile_size[0]-min_overlap)), math.ceil(image.shape[1] / (tile_size[1]-min_overlap)))
    
    step = (math.ceil((image.shape[0]-tile_size[0]) / num_tiles[0]), math.ceil((image.shape[1]-tile_size[1]) / num_tiles[1]))

    anchor_points = []

    for tile_row in range(num_tiles[0]+1):
        i = tile_row*step[0] if tile_row < num_tiles[0] else image.shape[0]-tile_size[0]
        # i -= image.shape[0]//2
        for tile_col in range(num_tiles[1]+1):
            j = tile_col*step[1] if tile_col < num_tiles[1] else image.shape[1]-tile_size[1]
            # j -= image.shape[1]//2
            anchor_points.append((i,j))

    return anchor_points



if __name__ == '__main__':
    image_file = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    image = plt.imread(image_file)
    tile_size = (768, 768)

    tiles = split_image(image)
    
    fig, ax = plt.subplots()

    ax.imshow(image, extent=[-image.shape[0]/2., image.shape[0]/2., -image.shape[1]/2., image.shape[1]/2. ])
    for xy in tiles:
        ax.add_patch(Rectangle(xy, width=tile_size[1], height=tile_size[0], linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()