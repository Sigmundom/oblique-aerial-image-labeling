import sys
import fiftyone as fo


def visualize(dataset):

    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=f'{dataset}/images',
        labels_path=f"{dataset}/annotations/instances_train.json",
        include_id=True,
    )

    # Verify that the class list for our dataset was imported
    print(coco_dataset.default_classes) 

    print(coco_dataset)

    session = fo.launch_app(coco_dataset)
    session.wait()


if __name__ == "__main__":
    dataset = sys.argv[1]
    visualize(dataset)