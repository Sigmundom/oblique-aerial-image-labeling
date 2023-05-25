from matplotlib import pyplot as plt
import numpy as np

def filter_height_values(points, mask_w, mask_h):
    gradient = np.zeros(len(points))
    neighbour_count = np.zeros(len(points), dtype=np.uint8)

    # Scale neighbourhood size based on mask size, because a more oblique view will contain more points per pixel.
    scale_x = mask_w / 500
    scale_y = mask_h / 500

    for i, point in enumerate(points):
        # Find the indices of the neighboring points
        neighbours = points[np.isclose(point[:2], points[:,:2], atol=(10*scale_y, 10*scale_x)).all(axis=1)]
        d_xy = np.linalg.norm(neighbours[:,:2], axis=1) - np.linalg.norm(point[2])
        d_z = np.mean(neighbours[:,2]) - point[2]

        neighbour_count[i] = len(neighbours)
        gradient[i] = (d_z/d_xy).sum()

    return points[(neighbour_count <= 5) | (gradient < 0)]




    # a = gradient > 0
    # color = np.empty_like(gradient, dtype=str)
    # color[a] = 'green'
    # color[np.invert(a)] = 'red'
    # color[neighbour_count <= 5] = 'black'

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(*points.T, color=color, s=1.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=110., azim=0)
    # plt.show()
    # plt.savefig('filter_points.png')