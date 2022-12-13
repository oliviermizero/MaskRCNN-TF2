"""
Mask R-CNN
Train on the maize segmentation dataset

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Adapted by Clay Christenson

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python maize.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python maize.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python maize.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate dectections
    python maize.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5> --split_num=number of splits
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == "__main__":
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

import datetime
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import skimage.io
import tensorflow as tf
from imgaug import augmenters as iaa
from PIL import Image
from sklearn import metrics

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/maize/")


############################################################
#  Configurations
############################################################


class MaizeConfig(Config):
    """Configuration for training on the maize segmentation dataset."""

    # Give the configuration a recognizable name
    NAME = "maize"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of class_ids (including background)
    NUM_CLASSES = 1 + 1  # Background + Kernel

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 1
    VALIDATION_STEPS = 1

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 3000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 170

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 2000

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 2000


class MaizeInferenceConfig(MaizeConfig):
    """Configuration for detection on the maize segmentation dataset."""

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################
VAL_IMAGE_IDS = []
LIST_IMAGE_IDS = []


class MaizeDataset(utils.Dataset):
    """ Used to create Maize Datasets, set the classes, and load the images and their masks

        Note:
        Set Validation and or List images IDS before running.
        Use LIST_IMAGE_IDS to load a dataset with a specfic set of images
        
        To add new classes simply uncomment the self.add_class(_,#,"name of class"), and change the name to whatever you need.
        
        Check and make sure the images, and their mask are displaying correctly before training.
    """

    def load_maize(self, dataset_dir, subset):

        # Add the class names using the base method from utils.Dataset
        """Implement way to add class_ids from dataset"""
        self.add_class("Maize", 1, "kernel")
        # self.add_class("Maize", 2, "cob")
        # self.add_class("Maize", 3, "aborted kernel")

        # Get filenames and annotation
        filenames = next(os.walk(dataset_dir))[2]
        annotation_filename = [
            filename for filename in filenames if filename.split(".")[1] == "json"
        ][0]

        # Which subset?
        # "val": Only images in VAL_IMAGE_IDS
        # "train": All images in dataset_dir except VAL
        # "test": All images in dataset_dir
        # "list": Only images in LIST_IMAGE_IDS
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test", "list"]
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        elif subset == "list":
            image_ids = LIST_IMAGE_IDS
        else:
            image_ids = [
                filename.split(".")[0]
                for filename in filenames
                if (
                    (not filename.split("_")[-1] in ["ground-truth.png", "label.png"])
                    and (filename.split(".")[1] == "png")
                )
            ]
        if subset == "train":
            image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        annotation_json = osp.join(dataset_dir, annotation_filename)
        with open(annotation_json) as f:
            annotations = json.load(f)

        # Add images
        for image_id in image_ids:
            image = image_id + ".png"
            # Get class ids
            class_ids = []
            for annotation in annotations:
                if annotation["image_name"] == image_id:
                    class_ids = annotation["class_ids"]

            self.add_image(
                "Maize",
                image_id=image_id,
                path=os.path.join(dataset_dir, image),
                class_ids=class_ids,
            )

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        
        This load_mask function converts the full label bitmap to the correct format for the model. 
        If you image labels are formatted in a different way you may need to adjust this to load the mask correctly.
        
        Bitmap file name must be image_name_label.png or image_name_label_ground-truth.png
        
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        image_name = info["id"]
        bitmap_path = str
        if osp.exists(osp.join((osp.dirname(info["path"])), f"{image_name}_label.png")):
            bitmap_path = osp.join(
                (osp.dirname(info["path"])), f"{image_name}_label.png"
            )
        elif osp.exists(
            osp.join(
                (osp.dirname(info["path"])), f"{image_name}_label_ground-truth.png"
            )
        ):
            bitmap_path = osp.join(
                (osp.dirname(info["path"])), f"{image_name}_label_ground-truth.png"
            )
        else:
            print(
                "Image labels must have the form *_label.png or *_label-ground-truth.png"
            )
            SystemExit()

        instances = info["class_ids"]
        # Read mask files from .png image
        mask = []
        class_ids = []
        bitmap = skimage.io.imread(bitmap_path)
        for instance in instances:
            if instance["class_id"] in self.class_ids:
                m = np.array(bitmap, np.uint32) == instance["id"]
                m.astype(bool)
                mask.append(m)
                class_ids.append(instance["class_id"])
            else:
                continue
        mask = np.stack(mask, axis=-1)
        class_ids = np.asarray(class_ids, dtype=np.int32)

        # Return mask, and array of class IDs of each instance.
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Maize":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


def train(model, config, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MaizeDataset()
    dataset_train.load_maize(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MaizeDataset()
    dataset_val.load_maize(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf(
        (0, 2),
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf(
                [iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]
            ),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0)),
        ],
    )

    # Implementation of early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    # *** This training schedule is an example. Update to your needs ***

    print("Train all layers")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=1000,
        augmentation=augmentation,
        layers="all",
        custom_callbacks=[early_stopping_callback],
    )


############################################################
#  Image spliting
#  This code is adapted from EarVision written by Cedar Warman
############################################################


def get_splits(image_width, split_number, overlap):
    # A really complicated fuction to get the split sections, with overlap
    image_splits = []
    total_image_width = image_width
    overlap_width = overlap

    if split_number == 1:
        image_splits.append([0, total_image_width])

    # This will be the most used case, as of now, with a split of 3 sub-images.
    # In this case, since a lot of the ear images have significant space on the
    # left and right, I want the center sub-image to not be too big. To avoid
    # this, I'll do the overlaps from the left and right images and leave the
    # center image unchanged.`
    elif split_number == 3:
        # Here's the split width if there's no overlap (note: probably will
        # need to do something about rounding errors here with certain image
        # widths).
        no_overlap_width = int(total_image_width / split_number)

        # Left split. The left side of the left split will always be zero.
        left_split = []
        left_split.append(0)

        # The other side of the left split will be the width (minus 1 to fix
        # the 0 index start) plus the overlap
        left_split.append(no_overlap_width + overlap_width)
        image_splits.append(left_split)

        # The middle has no overlap in this case
        middle_split = []
        middle_split.append(no_overlap_width - overlap)
        middle_split.append((no_overlap_width * 2) + overlap)
        image_splits.append(middle_split)

        # The right split is the opposite of the left split
        right_split = []
        right_split.append((2 * no_overlap_width) - overlap_width)
        right_split.append(total_image_width)
        image_splits.append(right_split)

    return image_splits


def spliting_image(image_np_array, split_list):
    # The fuction that actually splits the images
    array_list = []

    for split_nums in split_list:
        left_border = int(split_nums[0])
        right_border = int(split_nums[1])
        sub_array = image_np_array[:, left_border:right_border, :]
        array_list.append(sub_array)

    return array_list


def fix_relative_coord(output_dict, list_of_splits, image_position):
    output_dict_adj = output_dict

    # First we get a constant adjustment for the "image position". The
    # adjustment is where the left side of the current image starts, relative
    # to the entire image. We can get this from the list_of_splits.
    position_adjustment = list_of_splits[image_position][0]

    # Now we adjust the x coordinates of the 'rois' ndarray, We
    # don't need to adjust the y coordinates because we only split on the x. If
    # later I add splitting on y, then the y coordinates need to be adjusted.
    # This adjustment "shrinks" the relative coordinates down.
    adjusted_boxes = output_dict["rois"]

    # adjusted_boxes[:,[1,3]] = adjusted_boxes[:,[1,3]] *(split_width / image_width)

    # Adding the adjustment for which split image it is (the first image
    # doesn't need adjustment, hence the if statement).
    if image_position > 0:
        adjusted_boxes[:, [1, 3]] = adjusted_boxes[:, [1, 3]] + position_adjustment

    # Now adding back in the adjusted boxes to the original ndarray
    output_dict_adj["rois"] = adjusted_boxes

    return output_dict_adj


def pad_mask(results, list_of_splits, split_number, image_height):
    output_adj_dict = results
    # Getting the image width out of the list of splits (it's the right side of
    # the last split).
    image_width = list_of_splits[-1][1]
    height = image_height
    padded_masks = []
    r = results
    if split_number == 0:
        added_width = int(image_width - list_of_splits[0][1])
        padding_array = np.zeros([height, added_width])
        for i in range(len(r["masks"][1, 1, :])):
            combined = np.concatenate(
                (r["masks"][:, :, i].astype(np.uint8), padding_array), axis=1
            )
            padded_masks.append(combined)

    elif split_number == 1:
        added_width_l = int(list_of_splits[1][0])
        added_width_r = int(image_width - list_of_splits[1][1])
        padding_array_l = np.zeros([height, added_width_l])
        padding_array_r = np.zeros([height, added_width_r])
        for i in range(len(r["masks"][1, 1, :])):
            combined = np.concatenate(
                (
                    padding_array_r,
                    r["masks"][:, :, i].astype(np.uint8),
                    padding_array_l,
                ),
                axis=1,
            )
            padded_masks.append(combined)

    elif split_number == 2:
        added_width = int(list_of_splits[2][0])
        padding_array = np.zeros([height, added_width])
        for i in range(len(r["masks"][1, 1, :])):
            combined = np.concatenate(
                (padding_array, r["masks"][:, :, i].astype(np.uint8)), axis=1
            )
            padded_masks.append(combined)

    # Todo: Check shape because splits with no instance dectected have the wrong shape

    output_adj_dict["masks"] = padded_masks

    return output_adj_dict


def remove_edge_boxes(output_dict, list_of_splits, image_position):
    # This function removes the boxes that are near the edges of the splits, in an
    # attempt to remove boxes that are only a fraction of a seed.
    output_dict_rois_removed = output_dict
    # This is how close the edge of a box can be to the edge of the sub-image
    # before they get deleted
    edge_crop_width = 20
    image_width = list_of_splits[1][1]

    array_counter = 0
    delete_list = []

    # Pulling out the elements of the output_dict that will get modified
    adjusted_rois = output_dict["rois"]
    adjusted_scores = output_dict["scores"]
    adjusted_classes = output_dict["class_ids"]
    adjusted_masks = output_dict["masks"]

    # On the leftmost set of boxes, only ones near the right side will be
    # deleted
    if image_position == 0:
        # print("\n\nleft image")
        for box in adjusted_rois:
            xmax = box[3]
            if xmax > (image_width - edge_crop_width):
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        # print(len(delete_list))

        adjusted_rois = np.delete(adjusted_rois, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)
        adjusted_masks = np.delete(adjusted_masks, delete_list, 2)

    # Rightmost set
    elif image_position == (len(list_of_splits) - 1):
        # print("\n\nright image")
        for box in adjusted_rois:
            xmin = box[1]
            if xmin < edge_crop_width:
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        # print(len(delete_list))

        adjusted_rois = np.delete(adjusted_rois, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)
        adjusted_masks = np.delete(adjusted_masks, delete_list, 2)

    # All the middle sets
    else:
        # print("\n\nmiddle image")
        for box in adjusted_rois:
            xmax = box[3]
            xmin = box[1]
            if (xmin < edge_crop_width) or (xmax > (image_width - edge_crop_width)):
                # Adding the index to the list of indexes to be deleted
                delete_list.append(array_counter)
            array_counter += 1
        # print(len(delete_list))

        adjusted_rois = np.delete(adjusted_rois, delete_list, 0)
        adjusted_scores = np.delete(adjusted_scores, delete_list, 0)
        adjusted_classes = np.delete(adjusted_classes, delete_list, 0)
        adjusted_masks = np.delete(adjusted_masks, delete_list, 2)

    # Adding the modified arrays back into the output_dict
    # print("Original array length: ")
    # print(output_dict_rois_removed["rois"].shape[0])

    output_dict_rois_removed["rois"] = adjusted_rois
    output_dict_rois_removed["scores"] = adjusted_scores
    output_dict_rois_removed["class_ids"] = adjusted_classes
    output_dict_rois_removed["masks"] = adjusted_masks

    # print("Modified array length: ")
    # print(output_dict_rois_removed["rois"].shape[0])

    return output_dict_rois_removed


def do_non_max_suppression(results):
    # The actual nms comes from Tensorflow
    nms_vec_ndarray = utils.non_max_suppression(
        results["rois"], results["scores"], threshold=0.5
    )

    print("the length of the input array is: {}".format(len(results["rois"])))
    print("the length of nms ndarray is: {}".format(len(nms_vec_ndarray)))

    # Indexing the input dictionary with the output of non_max_suppression,
    # which is the list of boxes (and score, class) to keep.
    out_dic = results.copy()
    out_dic["rois"] = results["rois"][nms_vec_ndarray].copy()
    out_dic["scores"] = results["scores"][nms_vec_ndarray].copy()
    out_dic["class_ids"] = results["class_ids"][nms_vec_ndarray].copy()
    results["masks"] = np.transpose(results["masks"])
    out_dic["masks"] = results["masks"][nms_vec_ndarray].copy()
    out_dic["masks"] = np.transpose(out_dic["masks"])

    # Change to output dictionary
    return out_dic


############################################################
#  Bitmap
############################################################


def convert_to_bitmap(image, result):
    """Converts results to bitmap format and annotation format used to create datasets"""
    segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
    annotations = []
    counter = 1
    instances = result["masks"]
    for i in range(len(result["class_ids"])):
        class_id = int(result["class_ids"][i])
        instance_id = counter
        instance_mask = instances[:, :, i].astype(bool)
        segmentation_bitmap[instance_mask] = instance_id
        annotations.append({"id": instance_id, "class_id": class_id})
        counter += 1

    return segmentation_bitmap, annotations


############################################################
#  Detection
############################################################


def detect_splits(model, image, split_num):
    """Run detection of images splits and combine results"""

    # Split the Image with an overlap
    # Determine the subdividsion of the splits, based on number of splits wanted.
    # The splits will be a list of set of two numbers, the lower and upper bounds of the splits.
    splits = get_splits(image.shape[1], split_num, 50)

    # Actually split the image into the subdivision determined earlier
    split_images = spliting_image(image, splits)

    image_split_number = 0
    output_result = {"rois": [], "class_ids": [], "masks": np.array}
    ## Run Dectection on each Split
    for split_image in split_images:
        # Detect objects
        r = model.detect([split_image], verbose=0)[0]

        ## Fix relative coordinates
        adjusted_result = remove_edge_boxes(r, splits, image_split_number)
        adjusted_result = fix_relative_coord(r, splits, image_split_number)
        adjusted_result = pad_mask(
            adjusted_result, splits, image_split_number, image_height=image.shape[0]
        )
        ## Combine the predicted results
        if image_split_number == 0:
            output_result = adjusted_result
        else:
            output_result["rois"] = np.concatenate(
                (output_result["rois"], adjusted_result["rois"])
            )
            output_result["class_ids"] = np.concatenate(
                (output_result["class_ids"], adjusted_result["class_ids"])
            )
            output_result["scores"] = np.concatenate(
                (output_result["scores"], adjusted_result["scores"])
            )
            if len(adjusted_result["masks"]) == 0:
                continue
            output_result["masks"] = np.concatenate(
                (output_result["masks"], adjusted_result["masks"])
            )
        image_split_number += 1

    ## Remove redundant rois/mask
    output_result["masks"] = np.transpose(output_result["masks"])
    output_result["masks"] = np.swapaxes(output_result["masks"], 0, 1)
    output_result = do_non_max_suppression(output_result)

    return output_result


def detect(model, dataset_dir, subset, split_num):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MaizeDataset()
    dataset.load_maize(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    predictions_data = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        image_name = dataset.image_info[image_id]["id"]

        results = detect_splits(model, image, split_num)

        # Save predictions bitmap and annotation.json
        segmentation_bitmap, annotations = convert_to_bitmap(image, results)
        f = f"{image_name}_pred_label.png"
        f = osp.join(dataset_dir, f)
        Image.fromarray(segmentation_bitmap).save(f, "PNG")
        predictions_data.append({"image_name": image_name, "class_ids": annotations})

        # Save image with masks
        visualize.display_instances(
            image,
            results["rois"],
            results["masks"],
            results["class_ids"],
            dataset.class_names,
            results["scores"],
            show_bbox=False,
            show_mask=False,
            title="Predictions",
        )
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
        plt.close()

    # Save to json file
    file_name = os.path.join(dataset_dir, "predictions_annotations.json")
    with open(file_name, "w") as f:
        json.dump(predictions_data, f)

    print("Saved predictions_annotations.json to ", dataset_dir)


############################################################
#  Evaluate
############################################################
# TODO: Fix MaizeEval
class MaizeEval:
    def __init__(self) -> None:
        pass

    def _prepare(self):
        pass

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        pass

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        pass

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """
        pass


def build_maize_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format"""
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": mask,
            }
            results.append(result)
    return results


def evaluate_maize(model, dataset, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = detect_splits(model, image, split_num=3)
        t_prediction += time.time() - t

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_maize_results(
            dataset,
            coco_image_ids[i : i + 1],
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"].astype(np.uint8),
        )
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    # coco_results = coco.loadRes(results)

    # Evaluate
    maizeEval = MaizeEval()
    maizeEval.evaluate()
    maizeEval.accumulate()
    maizeEval.summarize()

    print(
        "Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)
        )
    )
    print("Total time: ", time.time() - t_start)


def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds):
    """Used to compute average recall"""
    AR = []
    for iou_threshold in list_iou_thresholds:

        try:
            recall, _ = utils.compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)

            AR.append(recall)

        except:
            AR.append(0.0)
            pass

    AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
    return AUC


def evaluate_model(
    model,
    dataset,
    config,
    limit=0,
    image_ids=None,
    type=str,
    split_num=3,
    list_iou_thresholds=None,
):
    """Evalute the model

    Args:
        limit (int, optional): To evaluate on a subset of the dataset. Defaults to the whole dataset.
        type (str, optional): Use "range" to evalute single image, Use "batch to evalute set of images.
    """
    assert type in ["range", "batch"]
    assert split_num in [1, 3]
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    if list_iou_thresholds is None:
        list_iou_thresholds = np.arange(0.5, 1.01, 0.1)

    if type == "range":
        image_id = np.random.choice(image_ids)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False
        )
        info = dataset.image_info[image_id]

        # Run object detection
        r = detect_splits(model, image, split_num)

        utils.compute_ap_range(
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            verbose=1,
        )

        visualize.display_differences(
            image,
            gt_bbox,
            gt_class_id,
            gt_mask,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
            dataset.class_names,
            show_box=False,
            show_mask=False,
            iou_threshold=0.5,
            score_threshold=0.5,
        )

    elif type == "batch":
        ARs = []
        APs = []
        for image_id in image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset, config, image_id, use_mini_mask=False
            )
            info = dataset.image_info[image_id]
            print(
                "image ID: {}.{} ({}) {}".format(
                    info["source"],
                    info["id"],
                    image_id,
                    dataset.image_reference(image_id),
                )
            )
            # Run object detection
            r = detect_splits(model, image, split_num)

            AP, precisions, recalls, overlaps = utils.compute_ap(
                gt_bbox,
                gt_class_id,
                gt_mask,
                r["rois"],
                r["class_ids"],
                r["scores"],
                r["masks"],
            )

            AR = compute_ar(r["rois"], gt_bbox, list_iou_thresholds)
            ARs.append(AR)
            APs.append(AP)

        mAP = np.mean(APs)
        mAR = np.mean(ARs)

        return mAP, mAR


############################################################
#  Command Line
############################################################


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Mask R-CNN for kernel counting and segmentation"
    )
    parser.add_argument("command", metavar="<command>", help="'train', 'detect'")
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/dataset/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default="../../logs",
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--subset",
        required=False,
        metavar="Dataset sub-directory",
        help="Subset of dataset to run prediction on",
    )
    parser.add_argument(
        "--split_num",
        required=False,
        default=1,
        type=int,
        metavar="Number of splits",
        help="Number of splits for prediction",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MaizeConfig()
    else:
        config = MaizeInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of class_ids
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset, args.split_num)
    else:
        print("'{}' is not recognized. " "Use 'train' or 'detect'".format(args.command))


if __name__ == "__main__":
    main()
