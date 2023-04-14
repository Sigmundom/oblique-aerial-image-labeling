import click
import fiftyone as fo


# def visualize_coco(dataset):

#     coco_dataset = fo.Dataset.from_dir(
#         dataset_type=fo.types.COCODetectionDataset,
#         data_path=f'{dataset}/images',
#         labels_path=f"{dataset}/annotations/segmentation.json",
#     )

#     # Verify that the class list for our dataset was imported
#     print(coco_dataset.default_classes) 

#     print(coco_dataset)

#     session = fo.launch_app(coco_dataset)
#     session.wait()

@click.command()
@click.argument('dataset')
def visualize_mask(dataset):

    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path=f'{dataset}/img',
        labels_path=f"{dataset}/label"
    )

    # Verify that the class list for our dataset was imported
    print(coco_dataset.default_classes) 

    print(coco_dataset)

    session = fo.launch_app(coco_dataset)
    session.wait()

if __name__ == "__main__":
    visualize_mask()