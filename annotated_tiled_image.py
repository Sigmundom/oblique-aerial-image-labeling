
from timeit import default_timer
from typing import List
from os import path
import json
import itertools
from PIL import Image
from matplotlib import cm, pyplot as plt
from rasterio.features import rasterize
import click
import numpy as np
import shapely.geometry as sg
from shapely.geos import TopologicalError
from shapely.validation import make_valid
from building import Building
from tiled_image import TiledImage, ensure_folder_exists
from utils import create_coco_rle_annotation, get_image_data

Image.MAX_IMAGE_PIXELS = 120000000

threshold_building_size = 500
threshold_building_part_size = 1000

class AnnotatedTiledImage(TiledImage):
    def __init__(
        self,
        cityjson,
        *args,
        image_id_start=0,
        annotation_id_start=0,
        **kwargs ,
        ):
        super().__init__(*args, **kwargs)
        self.buildings = self._extract_buildings_from_cityjson(cityjson)
        self.image_id_start = image_id_start
        self.annotation_id_start = annotation_id_start
        h, w = self.tile_size
        # Used for clipping polygons outside the tile when doing instance segmentation
        self.tile_polygon = sg.Polygon([(0,0), (0,w), (h,w), (h,0)]) 

    def get_buildings_in_tile(self, tile_index) -> List[Building]:
        ax, ay = self.anchors[tile_index]
        a_height, a_width = self.tile_size
        def is_building_in_tile(building):
            bx_min, by_min, bx_max, by_max = building.bbox
            return not (
                bx_max < ax or
                by_max < ay - a_height or 
                bx_min > ax + a_width or 
                by_min > ay
                )

        return list(filter(is_building_in_tile, self.buildings))

    def export_semantic_segmentation(self, annotation_format, label_walls):
        ensure_folder_exists(f'{self.output_folder}/images')
        if annotation_format == 'coco':
            ensure_folder_exists(f'{self.output_folder}/annotations')
        elif annotation_format == 'mask':
            ensure_folder_exists(f'{self.output_folder}/labels') 
        self.save_tile_data()

        annotations = []
        for tile_index in range(len(self)):
            buildings = self.get_buildings_in_tile(tile_index)
            if len(buildings) == 0:
                continue

            image_id = self.image_id_start + tile_index
            self.save_tile(tile_index)
            roofs = []
            walls = []
            for building in buildings:
                surfaces = [self.ic_to_tc(vertices_ic, tile_index) for vertices_ic in building.surface_vertices]
                for surface, surface_type in zip(surfaces, building.surface_types):
                    if surface_type == 1:
                        roofs.append(sg.Polygon(surface))
                    else: 
                        walls.append(sg.Polygon(surface))
            
            mask = np.zeros(self.tile_size, dtype=np.uint8)
            if label_walls:
                rasterize(walls, default_value=2, out_shape=self.tile_size, out=mask)
                rasterize(roofs, default_value=1, out_shape=self.tile_size, out=mask)
                wall_mask = mask == 2
                roof_mask = mask == 1
                if annotation_format=='coco':
                    annotations.append(create_coco_rle_annotation(image_id, 1, roof_mask))
                    annotations.append(create_coco_rle_annotation(image_id, 2, wall_mask))
            else:
                rasterize(walls, default_value=1, out_shape=self.tile_size, out=mask)
                rasterize(roofs, default_value=1, out_shape=self.tile_size, out=mask)
                if annotation_format=='coco':
                    annotations.append(create_coco_rle_annotation(image_id, 1, mask))

            if annotation_format == 'mask':
                plt.imsave(f'{self.output_folder}/labels/{self.image_name}_{tile_index}.png', mask, cmap=cm.gray)
            
       
        if annotation_format == 'coco':
            coco = self._create_annotation_base()
            if label_walls:
                coco['categories'] = [{
                        "id": 1,
                        "name": "Roof",
                    },
                    {
                        "id": 2,
                        "name": "Wall"
                    }]
            else:
                coco['categories'] = [{
                    "id": 1,
                    "name": "Building"
                }]
            coco['annotations'] = annotations
            with open(f'{self.output_folder}/annotations/segmentation.json', 'w', encoding='utf-8') as f:
                json.dump(coco, f)

    def export_instance_segmentation(self):
        coco = self._create_annotation_base()
        coco['categories'] = [
            {
                "id": 1,
                "name": "Building",
            }
        ]
        annotations = []
        annotation_id = self.annotation_id_start
        for tile_index in range(len(self)):
            image_id = self.image_id_start + tile_index
            buildings = self.get_buildings_in_tile(tile_index)
            if len(buildings) == 0:
                continue
            for building in buildings:
                surfaces = [self.ic_to_tc(vertices_ic, tile_index) for vertices_ic in building.surface_vertices]
                annotation = self._create_annotation(surfaces, image_id, annotation_id)
                if annotation is not None:
                    annotations.append(annotation)
                annotation_id += 1

        coco['annotations'] = annotations

        ensure_folder_exists(f'{self.output_folder}/annotations')
        with open(f'{self.output_folder}/annotations/instances_train.json', 'w', encoding='utf8') as f:
            json.dump(coco, f)

    def _create_annotation_base(self):
        return dict(
            info={},
            licenses=[],
            images=[
                {
                    "id": self.image_id_start + i,
                    "height": self.tile_size[0],
                    "width": self.tile_size[1],
                    "file_name": f'{self.image_name}_{i}.jpg',
                    "date_captured": str(self.get_date_captured())
                } for i in range(len(self))
            ],
        )

    def _create_annotation(self, surfaces, image_id, annotation_id):
        polygons = [sg.Polygon(surface) for surface in surfaces]
        
        for i in range(len(polygons)):
            if not polygons[i].is_valid:
                polygons[i] = make_valid(polygons[i])
                if not polygons[i].is_valid:
                    print('Still not valid')
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
                return None

        multi_poly = polygons[0]
        for i in range(1, len(polygons)):
            try:
                tmp = multi_poly.union(polygons[i])
                if isinstance(tmp, (sg.Polygon, sg.MultiPolygon)):
                    multi_poly = tmp
            except TopologicalError:
                print('Failed doing the union operation. Ignoring the issue.')

        multi_poly = multi_poly.simplify(1.0, preserve_topology=False)
        if multi_poly.area < threshold_building_size: return None 
        
        multi_poly = multi_poly.intersection(self.tile_polygon)
        if multi_poly.area < threshold_building_part_size: return None

        if isinstance(multi_poly, sg.Polygon):
            multi_poly = sg.MultiPolygon([multi_poly])

        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        segmentation = []
        for poly in multi_poly.geoms:
            if not isinstance(poly, sg.Polygon): continue
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

    def _extract_buildings_from_cityjson(self, cityjson) -> List[Building]:
        corner_points = ["UL", "UR", "LR", "LL"]
        x_coords = [float(self.image_data[f'{point}x']) for point in corner_points]
        y_coords = [float(self.image_data[f'{point}y']) for point in corner_points]
        x, x_max = min(x_coords), max(x_coords)
        y, y_max = min(y_coords), max(y_coords)

        all_buildings = cityjson['CityObjects']
        all_vertices = np.array(cityjson['vertices'])
        vertices_in_image = (all_vertices[:, 0] > x) & (all_vertices[:,0] < x_max) & (all_vertices[:,1] > y) & (all_vertices[:, 1] < y_max)
        buildings_in_image = []

        for building in all_buildings.values():
            geometry = building["geometry"]
            if len(geometry) > 1:
                print("Length is:", len(geometry))
            if len(geometry) == 0:
                continue
            boundaries = geometry[0]['boundaries']
            vertices_i = [v for boundary in boundaries for v in boundary[0]]
            
            if np.any(np.take(vertices_in_image, vertices_i, 0)):
                building = Building()
                surfaces = geometry[0]['semantics']['surfaces']
                draw = False
                for boundary, surface in zip(boundaries, surfaces):
                    vertices = np.take(all_vertices, boundary[0], axis=0)
                    vertices_ic = self.wc_to_ic(vertices)
                    if surface['type'] == 'WallSurface':
                        if np.ptp(vertices[:,2]) <= 0.75:
                            draw = True
                            surface_type = 3    
                        else: 
                            surface_type = 2
                    elif surface['type'] == 'RoofSurface':
                        surface_type = 1
                    else:
                        raise Exception(f'Surface type {surface_type} not recoginized')
                    # print(*vertices[:,2], surface_type)
                    building.add_surface(vertices_ic, surface_type)
                building.calculate_bbox()
                buildings_in_image.append(building)
                if draw:
                    building.draw(True)
                # print('###############################')

        
        return buildings_in_image

@click.command()
@click.argument('image_path')
@click.argument('seamline_path')
@click.option('-o', '--output-folder', default='outputs')
@click.option('-ts', '--tile-size', default=512) 
@click.option('-f','--annotation-format', default='mask', type=click.Choice(['mask', 'coco'], case_sensitive=False))
@click.option('-w', '--label-walls', is_flag=True)
def semantic_segmentation(image_path, seamline_path, output_folder, tile_size, annotation_format, label_walls):
    t_start = default_timer()
    print('Loading CityGML', end='')
    with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
        cityjson = json.load(f)
    print(' - Complete - Took', default_timer() - t_start, 'seconds' )

    t = default_timer()
    image_name = path.basename(image_path).split('.')[0]
    print('Processing ', image_name, '...')
    image = Image.open(image_path)
    image_data = get_image_data(seamline_path, image_name)
    
    tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(tile_size, tile_size))

    print('Exporting semantic segmentation', end=' - ')
    t = default_timer()
    tiled_image.export_semantic_segmentation(annotation_format, label_walls)
    print(default_timer() -t, 'seconds')


# if __name__ == '__main__':
#     configs = [
#         # dict(
#         #     image_path = 'images/Bakoverrettede bilder/30196_127_02033_210427_Cam4B.jpg',
#         #     seamline_path = 'images/Somlinjefiler/cam4B.sos',
#         #     output_folder = 'outputs/back'
#         # ),
#         # dict(
#         #     image_path = 'images/Framoverrettede bilder/30196_127_02023_210427_Cam7F.jpg',
#         #     seamline_path = 'images/Somlinjefiler/cam7F.sos',
#         #     output_folder = 'outputs/front'
#         # ),
#         # dict(
#         #     image_path = 'images/Høyrerettede bilder/30196_124_01897_210427_Cam5R.jpg',
#         #     seamline_path = 'images/Somlinjefiler/cam5R.sos',
#         #     output_folder = 'outputs/right'   
#         # ),
#         # dict(
#         #     image_path = 'images/Venstrerettede bilder/30196_128_02062_210427_Cam6L.jpg',
#         #     seamline_path = 'images/Somlinjefiler/cam6L.sos',
#         #     output_folder = 'outputs/left'   
#         # ),
#         dict(
#             image_path = 'images/Vertikalbilder/30196_127_02029_210427_Cam0N.jpg',
#             seamline_path = 'images/Somlinjefiler/cam0N.sos',
#             output_folder = 'outputs/vertikal/512'   
#         )
#     ]
#     t_start = default_timer()
#     print('Loading CityGML', end='')
#     with open('3DBYGG_BASISDATA_4202_GRIMSTAD_5972_FKB-BYGNING_SOSI_CityGML_reprojected.json', encoding='utf8') as f:
#         cityjson = json.load(f)
#     print(' - Complete - Took', default_timer() - t_start, 'seconds' )

#     for c in configs:
#         t = default_timer()
#         image_path = c['image_path']
#         seamline_path = c['seamline_path']
#         output_folder = c['output_folder']
#         image_name = path.basename(image_path).split('.')[0]
#         print('Processing ', image_name, '...')
#         image = Image.open(image_path)
#         image_data = get_image_data(seamline_path, image_name)
#         print('Reading seamline file took', default_timer()-t, 'seconds')
#         tiled_image = AnnotatedTiledImage(cityjson, image, image_name, image_data, output_folder=output_folder, tile_size=(512, 512))

#         print('Exporting images', end=' - ')
#         t = default_timer()
#         tiled_image.export_image_tiles()
#         print(default_timer() - t, 'seconds')
#         print('Exporting semantic segmentation', end=' - ')
#         # t = default_timer()
#         # tiled_image.export_semantic_segmentation()
#         # print(default_timer() -t, 'seconds')
#         # print('Exporting instance segmentation')
#         t = default_timer()
#         tiled_image.export_instance_segmentation()
#         print(default_timer() -t, 'seconds')

#     total_time = default_timer() - t_start
#     print(f'Whole process took {total_time} seconds for {len(configs)} images.\n That is {total_time/len(configs)} seconds per image')