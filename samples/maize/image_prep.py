"""
Image and Label prep. Labels are currently stored in Segment.ai.

This script will download and split the images and labels, and prepare the annotations for Mask_RCNN
"""
import json
import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from segments import SegmentsClient, SegmentsDataset
from skimage.measure import regionprops


def write_annotations(dataset, output_dir, classes):
    """ Creates the annotation file for the full images
        
    output json file formatt
    [{'image_name': 1645_3, class_ids: [{'id': 1, 'class_id': 1},...,{'id': 343, 'class_id': 1}]},...,{"image_name":...}]
    """
    json_data = []
    for sample in dataset:
        image_name = sample["file_name"].split(".")[0]
        class_ids = []
        for instance in sample["annotations"]:
            if str(instance.get("category_id")) in classes:
                class_ids.append(
                    {"id": instance.get("id"), "class_id": instance.get("category_id")}
                )

        annotation = {"image_name": image_name, "class_ids": class_ids}
        json_data.append(annotation)

    file_name = os.path.join(output_dir, "full_annotations.json")  # rename
    with open(file_name, "w") as f:
        json.dump(json_data, f)

    print("Exported to {}".format(file_name))


def save_split_images(split_image, output_dir, image_name):
    """Saves splits images"""
    counter = 1
    split_iname = image_name.split("_")
    for image in split_image:
        if len(split_iname) == 4:
            output_name = "{}_{}_s{}_{}.png".format(
                split_iname[0], split_iname[1], str(counter), split_iname[2]
            )
        elif len(split_iname) == 5:
            output_name = "{}_{}_{}_s{}_{}.png".format(
                split_iname[0],
                split_iname[1],
                split_iname[2],
                str(counter),
                split_iname[3],
            )
        else:
            output_name = "{}_s{}.png".format(image_name, str(counter))
        image = Image.fromarray(image)
        output_string = osp.join(output_dir, output_name)
        image.save(output_string)
        counter += 1


def split_annotations(split_name, split_label_array, annotation_json, output_dir):
    """ Creates annotations for the split image. """

    regions = regionprops(split_label_array)
    regions = [region.label for region in regions]

    with open(annotation_json) as f:
        annotations = json.load(f)

    class_ids = []
    image_name = split_name.split("_s")[0]
    for annotation in annotations:
        if annotation.get("image_name") == image_name:
            for class_id in annotation["class_ids"]:
                if class_id.get("id") in regions:
                    class_ids.append(
                        {"id": class_id.get("id"), "class_id": class_id.get("class_id")}
                    )
    if len(class_ids) == 0:
        print(f"No labels found on {split_name}")
        return
    else:
        split_annotations = {"image_name": split_name, "class_ids": class_ids}

    file_name = os.path.join(output_dir, "split_annotations.json")
    if osp.exists(file_name):
        with open(file_name, "r+") as f:
            json_data = json.load(f)
    else:
        json_data = []
    with open(file_name, "w") as f:
        json_data.append(split_annotations)
        json.dump(json_data, f)

    print("Added {} annotation to {}".format(split_name, file_name))


def remove_empty_images(image_dir):
    """_summary_

    Args:
        image_dir (_type_): _description_
    """
    filenames = next(os.walk(image_dir))[2]

    annotation_filename = [
        filename for filename in filenames if filename.split(".")[1] == "json"
    ][0]

    with open(osp.join(image_dir, annotation_filename), "r") as f:
        annotations = json.load(f)

    image_ids = [
        filename
        for filename in filenames
        if (
            (not filename.split("_")[-1] == "label.png")
            and (filename.split(".")[1] == "png")
        )
    ]

    annotated_images = [annotation["image_name"] for annotation in annotations]
    for image_id in image_ids:
        if image_id.split(".")[0] in annotated_images:
            continue
        else:
            os.remove(osp.join(image_dir, image_id))
            label_id = "{}_label.png".format(image_id.split(".")[0])
            os.remove(osp.join(image_dir, label_id))


def main():

    parser = ArgumentParser()

    parser.add_argument(
        "api_key", help="personal api_key for Segments.ai", type=str, action="store",
    )

    parser.add_argument(
        "release_version",
        help="release version to download from Segments.ai",
        type=str,
        action="store",
    )

    parser.add_argument(
        "filter",
        choices=["labeled", "unlabeled", "reviewed"],
        help="filter for Segments.ai",
        type=str,
        action="store",
    )

    parser.add_argument(
        "-d",
        "--dataset_name",
        dest="dataset_name",
        help="dataset_name to download from Segments.ai",
        default="Cchristenson3/Mazie_Images",
        type=str,
        action="store",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="location to store images and annotations.",
        default="./full",
        type=str,
        action="store",
    )

    parser.add_argument(
        "-c",
        "--classes",
        dest="classes",
        help="classes you want to keep annotations from.",
        default="[1]",
        type=list,
        action="store",
    )

    args = parser.parse_args()

    def bailout():
        parser.print_help()
        raise SystemExit

    if not args.release_version:
        bailout()

    client = SegmentsClient(args.api_key)
    dataset_name = args.dataset_name
    release_version = args.release_version
    release = client.get_release(dataset_name, release_version)
    dataset = SegmentsDataset(
        release, filter_by=args.filter, segments_dir=args.output_dir
    )
    if args.filter in ["labeled", "reviewed"]:
        dataset_folder = args.dataset_name.replace("/", "_")
        img_dir = osp.join(args.output_dir, dataset_folder, args.release_version)
        write_annotations(dataset, img_dir, classes=args.classes)

        split_output_dir = osp.join(osp.dirname(args.output_dir), "split")
        if not osp.exists(split_output_dir):
            os.makedirs(split_output_dir)

        assert (
            not img_dir is split_output_dir
        ), "The image directory and the output directory must be different"
        annotation_json = osp.join(img_dir, "full_annotations.json")
        assert osp.exists(
            annotation_json
        ), f"{annotation_json} not found. Annotation file must be in the image dir and have name 'full_annotations'"

        print("\nProcessing images...")
        image_ids = next(os.walk(img_dir))[2]
        for image_id in image_ids:
            if len(image_id.split("_")) > 3 or image_id.split(".")[1] == "json":
                continue
            image_name = image_id.split(".")[0]
            image_path = osp.join(img_dir, image_id)
            label_name = image_name + "_label_ground-truth.png"
            label_path = osp.join(img_dir, label_name)

            # Opened image and label and convert them to numpy array
            with Image.open(image_path) as image_data:
                image_array = np.array(image_data)
            with Image.open(label_path) as label_data:
                label_array = np.array(label_data)

            assert (
                image_array[:, :, 1].shape == label_array.shape
            ), f"{image_array.shape} is not equal to {label_array.shape}"

            # Split image and label
            split_image = np.array_split(image_array, 3, axis=1)
            split_label = np.array_split(label_array, 3, axis=1)

            # Save Splits
            save_split_images(split_image, split_output_dir, image_name)
            save_split_images(split_label, split_output_dir, label_name.split(".")[0])

            # Generate class annotations for the label images
            for split_n, split in enumerate(split_label):
                split_name = "{}_s{}".format(image_name, str(split_n + 1))
                split_annotations(split_name, split, annotation_json, split_output_dir)

        remove_empty_images(split_output_dir)


if __name__ == "__main__":
    main()
