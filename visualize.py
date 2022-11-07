import fiftyone as fo
import fiftyone.zoo as foz

coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path='dataset/images/train',
    labels_path="dataset/annotations/instance_train.json",
    include_id=True,
)

# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]

print(coco_dataset)

# dataset = foz.load_zoo_dataset("quickstart")
# dataset = fo.Dataset("dataset")
# session = fo.launch_app(dataset)