from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from utils.make_histogram import make_histogram

def filter_height_values(points, mask_w, mask_h, cropbox, cropbox_size):
    d_z = np.zeros(len(points))
    neighbour_count = np.zeros(len(points), dtype=np.uint8)

    # Scale neighbourhood size based on mask size, because a more oblique view will contain more points per pixel.
    scale_x = mask_w / 500
    scale_y = mask_h / 500

    # ax = fig.add_subplot(111)
    # plt.close()
    # fig = plt.figure(tight_layout=True)
    # ax = plt.gca()
    # image = Image.open('analysis/test_grimstad/476130_6465900/images/30196_127_02034_210427_Cam4B.jpg').resize((cropbox_size, cropbox_size))
    # image = image.crop(cropbox)
    # im = np.array(image)
    # print(points[:,0].min(), points[:,0].max())
    # print(points[:,1].min(), points[:,1].max())
    # print(mask_w, mask_h)
    # print(im.shape)
    # ax.set_xlim(0, mask_w)
    # ax.set_ylim(0, mask_h)
    # ax.set_xbound(0, mask_w)
    # ax.set_ybound(0, mask_h)
    # ax.imshow(im, alpha=0.7, extent=(0, mask_w, 0, mask_h), interpolation='nearest')
    # ax.scatter(points[:,1], mask_h - points[:,0], c=points[:,2], s=5)
    # plt.axis('off')
    # plt.savefig('height_points.png', bbox_inches='tight')
    # exit()
    for i, point in enumerate(points):
        # Find the indices of the neighboring points
        neighbours = points[np.isclose(point[:2], points[:,:2], atol=(10*scale_y, 10*scale_x)).all(axis=1)]
        # d_xy = np.linalg.norm(neighbours[:,:2] - point[2], axis=1)
        # d_y = neighbours[:,0] - point[0]
        # d_z = neighbours[:,2] - point[2]
        neighbour_count[i] = len(neighbours)
        d_z[i] = np.mean(neighbours[:,2] - point[2])

        # d_z[i] = (d_z/d_xy).sum()
    # res = points[(neighbour_count <= 5) | (d_z < 0)]
    # make_histogram(d_z, 20, 'gradient')
    # res = points[(d_z < 0)]
    # plt.close()
    # fig = plt.figure(tight_layout=True)
    # ax = plt.gca()
    # image = Image.open('analysis/test_grimstad/476130_6465900/images/30196_127_02034_210427_Cam4B.jpg').resize((cropbox_size, cropbox_size))
    # image = image.crop(cropbox)
    # im = np.array(image)
    # print(res[:,0].min(), res[:,0].max())
    # print(res[:,1].min(), res[:,1].max())
    # print(mask_w, mask_h)
    # print(im.shape)
    # ax.set_xlim(0, mask_w)
    # ax.set_ylim(0, mask_h)
    # ax.set_xbound(0, mask_w)
    # ax.set_ybound(0, mask_h)
    # ax.imshow(im, alpha=0.7, extent=(0, mask_w, 0, mask_h), interpolation='nearest')
    # ax.scatter(res[:,1], mask_h - res[:,0], c=res[:,2], s=5)
    # plt.axis('off')
    # plt.savefig('height_points_res.png', bbox_inches='tight')
    return points[(neighbour_count <= 5) | (d_z < 0)]




    # a = gradient > 0
    # color = np.empty_like(gradient, dtype=str)
    # color[a] = 'green'
    # color[np.invert(a)] = 'red'
    # color[neighbour_count <= 5] = 'black'
