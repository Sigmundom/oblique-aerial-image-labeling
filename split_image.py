import math
import PIL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import tile_size, min_overlap


def split_image(im: PIL.Image):
    num_tiles = (math.ceil(im.height / (tile_size[0]-min_overlap)), math.ceil(im.width / (tile_size[1]-min_overlap)))
    
    step = (math.ceil((im.height-tile_size[0]) / num_tiles[0]), math.ceil((im.width-tile_size[1]) / num_tiles[1]))

    anchor_points = []

    for tile_row in range(num_tiles[0]+1):
        i = tile_row*step[0] if tile_row < num_tiles[0] else im.height-tile_size[0]
        i -= im.height//2
        for tile_col in range(num_tiles[1]+1):
            j = tile_col*step[1] if tile_col < num_tiles[1] else im.width-tile_size[1]
            j -= im.width//2
            anchor_points.append((i,j))

    return anchor_points

def get_tile(im: PIL.Image, anchor):
    im_w, im_h = im.size
    h, w = tile_size
    r, c = anchor
    r += im_h//2
    c += im_w//2
    box = (c, im_h-r-h, c+w, im_h-r)

    return im.crop(box)



if __name__ == '__main__':
    image_file = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg'
    image = PIL.Image.open(image_file)

    tiles = split_image(image)
    
    fig, ax = plt.subplots()

    ax.imshow(image, extent=[-image.height/2., image.height/2., -image.width/2., image.width/2. ])
    for xy in tiles:
        ax.add_patch(Rectangle(xy, width=tile_size[1], height=tile_size[0], linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()