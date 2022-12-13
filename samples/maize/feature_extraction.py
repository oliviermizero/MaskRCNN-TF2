"""
Feature Extraction 
"""
from argparse import ArgumentParser

import mrcnn.visualize as visualize
import numpy as np

from maize import MaizeDataset


def percent_ear_fill(image_masks, class_ids):
    """Calculate ear fill given the image_masks and class_ids"""
    total_masked = np.sum(image_masks)
    kernel_masked = 0
    for image_mask, class_id in zip(image_masks, class_ids):
        if class_id == 1:
            kernel_masked += np.sum(image_mask)

    percent_ear_fill = (kernel_masked / total_masked) * 100
    return percent_ear_fill


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


# TODO: Implement new Method of calulating relative kernel location
def build_location_index(bboxes, masks):
    """Builds list of kernel ids sorted based on location top left->top right"""
    location_array = []
    for id, bbox in enumerate(bboxes):
        x, y = bbox[0], bbox[1]
        location_array.append([id, x, y, bbox, masks[:, :, id]])
    slocation_array = sorted(location_array, key=lambda k: [k[1], k[2]])
    position_id = 1
    location_index = []
    for kernel in slocation_array:
        kernel_dict = {
            "id": kernel[0],
            "position_id": position_id,
            "bbox": kernel[3],
            "mask": kernel[4],
        }
        location_index.append(kernel_dict)
        position_id += 1

    return location_index


def get_row_count(location_index):
    """Determine the number of kernels per row and the number of rows"""
    updated_index = []
    kernel_per_row = []
    st_bbox = location_index[0]["bbox"]
    row_num = 1
    kernel_num = 1
    for kernel in location_index:
        bbox = kernel["bbox"]
        mid_point = int((bbox[0] + bbox[2]) / 2)
        if mid_point in range(st_bbox[0], st_bbox[2]):
            kernel.update({"row": row_num})
            kernel_num += 1
        else:
            row_num += 1
            st_bbox = kernel["bbox"]
            kernel_per_row.append(kernel_num)
            kernel_num = 1
            kernel.update({"row": row_num})

        updated_index.append(kernel)

    return row_num, kernel_per_row, updated_index


def main():

    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_dir",
        dest="dataset_dir",
        help="location of images and annotations.",
        type=str,
        action="store",
    )
    parser.add_argument(
        "-s",
        "--subset",
        dest="subset",
        help="Subset of dataset to use.",
        type=str,
        action="store",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="Location to store cvs of features",
        type=str,
        action="store",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    dataset = MaizeDataset()
    dataset.load_maize(dataset_dir, args.subset)
    dataset.prepare()

    generated_data = []
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        image_name = dataset.image_info[image_id]["id"]
        masks, class_ids = dataset.load_mask(image_id)

        ear_fill = percent_ear_fill(masks, class_ids)
        masks = masks.swapaxes(0, 2)

        kernel_masks = np.array(
            [mask for mask, class_id in zip(masks, class_ids) if class_id == 1]
        )
        masks = masks.swapaxes(0, 2)
        kernel_masks = kernel_masks.transpose()

        kernel_masks = kernel_masks.transpose()
        kernel_masks = kernel_masks.swapaxes(0, 1)

        kernel_count = kernel_masks.shape[2]

        kernel_bboxes = extract_bboxes(kernel_masks)

        row_index = build_location_index(kernel_bboxes, kernel_masks)

        row_count, kernel_per_row, u_location_index = get_row_count(row_index)
        column_count = np.max(kernel_per_row)

        generated_data.append(
            {
                "Image name": image_name,
                "Percent Ear Fill": ear_fill,
                "Number of Kernels": kernel_count,
                "Number of Rows": row_count,
                "Number of Kernels Per Row": column_count,
            }
        )
        print(
            f"Image name: {image_name} \n Percent Ear Fill: {ear_fill} \n Number of Kernels: {kernel_count} \n Number of Rows: {row_count} \n Number of Kernels Per Row: {column_count}"
        )
        visualize.display_top_masks(
            image, masks, class_ids, dataset.class_names, limit=2
        )


if __name__ == "__main__":
    main()
