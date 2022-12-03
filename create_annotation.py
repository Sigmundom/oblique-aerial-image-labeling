import itertools
import PIL
from matplotlib.patches import Rectangle
import numpy as np
import shapely.geometry as sg
from shapely.geos import TopologicalError
from shapely.validation import make_valid
import matplotlib.pyplot as plt
import descartes
import config

h, w = config.tile_size
tile = sg.Polygon([(0,0), (0,w), (h,w), (h,0)])

def create_annotation(surfaces, image_id, annotation_id):
    polygons = [sg.Polygon(surface) for surface in surfaces]
    
    for i in range(len(polygons)):
        if not polygons[i].is_valid:
            polygons[i] = make_valid(polygons[i])
            if not polygons[i].is_valid:
                print('Still not valid')
                np.save(f'not_valid_{image_id}_{annotation_id}', surfaces)
                return None
        if isinstance(polygons[i], sg.GeometryCollection):
            for p in polygons[i].geoms:
                if isinstance(p, sg.Polygon):
                    polygons[i] = p
                    break
        if not polygons[i].is_closed:
            polygons[i] = polygons[i].buffer(1).buffer(-1)
            if not polygons[i].is_closed:
                print('Still not closed')
                np.save(f'not_closed_{image_id}_{annotation_id}', surfaces)
            return None

    multi_poly = polygons[0]
    for i in range(1, len(polygons)):
        try:
            tmp = multi_poly.union(polygons[i])
            if isinstance(tmp, (sg.Polygon, sg.MultiPolygon)):
                multi_poly = tmp
        except TopologicalError:
            print('Failed doing the union operation. Ignoring the issue.')
            np.save(f'wierd_{image_id}_{annotation_id}', surfaces)

    multi_poly = multi_poly.simplify(1.0, preserve_topology=False)
    if multi_poly.area < config.threshold_building_size: return None 
    
    multi_poly = multi_poly.intersection(tile)
    if multi_poly.area < config.threshold_building_part_size: return None

    if isinstance(multi_poly, sg.Polygon):
        multi_poly = sg.MultiPolygon([multi_poly])

    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    segmentation = []
    for poly in multi_poly.geoms:
        segmentation.append([x for x in itertools.chain.from_iterable(itertools.zip_longest(*poly.exterior.coords.xy))])
    
    
    return {
        'segmentation': segmentation,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': 1,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

if __name__=='__main__':    
    # all_s = np.load('surfaces.npy', allow_pickle=True)
    # im = PIL.Image.open('tile.jpg')
    all_s = [np.load('not_closed_62_592.npy', allow_pickle=True)]
    im = PIL.Image.open('dataset/images/train/30196_127_02033_210427_Cam4B_567_-1745.jpg')
    np_im = np.asarray(im)
    ax = plt.gca()
    ax.imshow(np_im)
    for s in all_s:
        annotation = create_annotation(s, 0, 0)
        if annotation is None: continue
        segments = annotation['segmentation']
        polygons = []
        for s in segments:
            xy = []
            for i in range(0, len(s), 2):
                xy.append((s[i], s[i+1]))
            polygons.append(sg.Polygon(xy))
        
            
        for poly in polygons:
            ax.add_patch(descartes.PolygonPatch(poly, ec='k', alpha=0.5))
        xx, yy, w, h = annotation['bbox']
        ax.add_patch(Rectangle((xx,yy), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        
    plt.show()

