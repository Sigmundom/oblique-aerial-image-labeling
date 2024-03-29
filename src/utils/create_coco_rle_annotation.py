import numpy as np
from pycocotools.mask import encode, area, toBbox

def create_coco_rle_annotation(image_id, category_id, mask):
    create_coco_rle_annotation.id_counter += 1
    rle = encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return {
        'image_id': image_id,
        'category_id': category_id,
        'segmentation': rle,
        'id': create_coco_rle_annotation.id_counter,
        'area': int(area(rle)),
        'bbox': toBbox(rle).tolist(),
    }
create_coco_rle_annotation.id_counter = 0
