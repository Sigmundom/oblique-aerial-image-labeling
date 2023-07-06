import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
import json

def png_to_coco(png_path, output_json_path):
    # Load PNG mask
    mask = io.imread(png_path)
    if mask.ndim != 2:
        raise ValueError("Invalid PNG mask. Must be a grayscale image.")

    # Convert PNG mask to RLE mask
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to str

    # Create COCO annotation
    annotation = {
        "id": 1,  # Replace with your desired annotation ID
        "image_id": 1,  # Replace with the corresponding image ID
        "category_id": 1,  # Replace with the corresponding category ID
        "segmentation": rle,
        "area": int(mask.sum()),
        "bbox": mask_utils.toBbox(rle).tolist(),
        "iscrowd": 0  # Set to 1 if the mask represents a crowd region
    }

    # Create COCO annotation file
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "width": 1400,
                "height": 1700,
                "file_name": "img.jpeg"
            }
        ],
        "annotations": [annotation],
        "categories": []  # Replace with your category information
    }

    with open(output_json_path, "w") as output_file:
        json.dump(coco_data, output_file)

    print("Conversion completed successfully.")

# Usage example
png_path = "ground_truth_2.png"
output_json_path = "output.json"
png_to_coco(png_path, output_json_path)
