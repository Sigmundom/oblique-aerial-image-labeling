import fiftyone as fo

coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path='dataset/images/train',
    labels_path="dataset/annotations/instances_train.json",
    include_id=True,
)

# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes) 

print(coco_dataset)

session = fo.launch_app(coco_dataset)
session.wait()