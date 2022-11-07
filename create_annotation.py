import itertools
import numpy as np
import shapely.geometry as sg
from functools import reduce
import matplotlib.pyplot as plt
import descartes

def visualize(polygon):
    ax = plt.gca()
    x, y, max_x, max_y = polygon.bounds
    ax.set_xlim(x, max_x); ax.set_ylim(y, max_y)
    ax.set_aspect('equal')
    
    # ax.add_patch(descartes.PolygonPatch(wall, ec='k', alpha=0.5))

    ax.add_patch(descartes.PolygonPatch(polygon, ec='green', fc='red', alpha=0.5))
    
    plt.show()

def union(a,b):
    return a.union(b)

# h, w = config.tile_size
# tile = sg.Polygon([(0,0), (0,w), (h,w), (h,0)])

def create_annotation(surfaces, image_id, annotation_id):
    polygons = [sg.Polygon(r) for r in surfaces]
    
    for p in polygons:
        if not p.is_valid:
            print('Not valid!')
            # visualize(p)
            # visualize(sg.MultiPolygon(polygons))
            return None
        if not p.is_closed:
            print('Not closed')
            # visualize(p)
            # visualize(sg.MultiPolygon(polygons))
            return None


    multi_poly = reduce(union, polygons).simplify(1)
    # multi_poly = multi_poly.intersection(tile)

    if type(multi_poly) == sg.Polygon:
        multi_poly = sg.MultiPolygon([multi_poly])
    if type(multi_poly) == sg.LineString:
        print('LineString!')
        # visualize(multi_poly)

    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    segmentation = []
    for poly in multi_poly:
        if type(poly) == sg.LineString:
            print('Actually a linestring!')
            print(type(multi_poly))
            # visualize(poly)
        else :
            segmentation.append([x for x in itertools.chain.from_iterable(itertools.zip_longest(*poly.exterior.coords.xy)) if x ])
    
    


    return {
        'segmentation': segmentation,
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': 1,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }   


# def draw_mask(walls, roofs):
#     x_min = walls[:,:,0].min()
#     x_max = walls[:,:,0].max()
#     y_min = walls[:,:,1].min()
#     y_max = walls[:,:,1].max()

#     roof_polygons = [sg.Polygon(r) for r in roofs]
#     wall_polygons = [sg.Polygon(w) for w in walls]

#     roof = reduce(union, roof_polygons)
#     dilated = roof.buffer(0.5)
#     roof = dilated.buffer(-0.5).simplify(1)

#     wall = reduce(union, wall_polygons)
#     wall = wall.difference(roof)
#     eroded = wall.buffer(-0.5)
#     wall = eroded.buffer(0.5).simplify(1)

#     # print(roof.coords)
#     print(type(roof))
#     print(type(wall))
#     exit()
#     x,y = roof.exterior.coords.xy
#     print(len(x), len(y))
#     # exit()
#     print([w.exterior.coords.xy for w in wall])
#     # print('######')
#     # print(wall.coords)
#     # exit()
#     ax = plt.gca()
#     ax.set_xlim(x_min-3, x_max+3); ax.set_ylim(y_min-3, y_max+3)
#     ax.set_aspect('equal')
    
#     # ax.add_patch(descartes.PolygonPatch(wall, ec='k', alpha=0.5))

#     ax.add_patch(descartes.PolygonPatch(roof, ec='green', fc='red', alpha=0.5))
    
#     plt.show()

if __name__=='__main__':
    # image_file = 'images/Framoverrettede bilder/30196_127_02026_210427_Cam7F.jpg'
    # image_name = path.basename(image_file).split('.')[0]
    # image_data = get_image_data(image_name)
    # wc_to_ic = get_wc_to_ic_transformer(image_data)

    # with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json') as f:
    #     city_json = json.load(f)

    # all_buildings = city_json['CityObjects']
    # all_vertices = np.array(city_json['vertices'])
    # buildings = get_buildings_in_image(all_buildings, all_vertices, image_data)
    # it = iter(buildings.values())
    # next(it)
    # building = next(it)
    

    # geometry = building['geometry'][0]

    # boundaries = geometry['boundaries']

    # surfaces = []

    # for boundary in boundaries:
    #     surfaces.append([wc_to_ic(*all_vertices[v_i]) for v_i in boundary[0]])

    # surfaces = np.array(surfaces)
    # np.save('surfaces', surfaces)
    
    surfaces = np.load('surfaces.npy', allow_pickle=True)
    annotation = create_annotation(surfaces, 0, 0)
    print(annotation)


    # surfaces = geometry['semantics']['surfaces']
    # walls = []
    # roofs = []

    # for boundary, surface in zip(boundaries, surfaces):
    #     vertices = [wc_to_ic(*all_vertices[v_i]) for v_i in boundary[0]]
        # if surface['type'] == 'RoofSurface':
        #     roofs.append(vertices)
        # elif surface['type'] == 'WallSurface':
        #     walls.append(vertices)
        # else:
        #     print('New surface type!', surface['type'])

    # walls = np.array(walls)
    # roofs = np.array(roofs)
    # np.save('walls', walls)
    # np.save('roofs', roofs)

    # walls = np.load('walls.npy')
    # roofs = np.load('roofs.npy', allow_pickle=True)

    # draw_mask(walls, roofs)