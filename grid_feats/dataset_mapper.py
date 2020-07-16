# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)


def annotations_to_instances_with_attributes(
    annos, image_size, mask_format="polygon", load_attributes=False, max_attr_per_ins=16
):
    """
    Extend the function annotations_to_instances() to support attributes
    """
    boxes = [
        BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
        for obj in annos
    ]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert (
                        segm.ndim == 2
                    ), "Expect segmentation of 2 dimensions, got {}.".format(segm.ndim)
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    if len(annos) and load_attributes:
        attributes = -torch.ones((len(annos), max_attr_per_ins), dtype=torch.int64)
        for idx, anno in enumerate(annos):
            if "attribute_ids" in anno:
                for jdx, attr_id in enumerate(anno["attribute_ids"]):
                    attributes[idx, jdx] = attr_id
        target.gt_attributes = attributes

    return target


class AttributeDatasetMapper(DatasetMapper):
    """
    Extend DatasetMapper to support attributes.
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # fmt: off
        self.use_attribute      = cfg.MODEL.ATTRIBUTE_ON
        self.max_attr_per_ins  = cfg.INPUT.MAX_ATTR_PER_INS
        # fmt: on

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # if "annotations" not in dataset_dict:
        #     image, transforms = T.apply_augmentations(
        #         ([self.crop] if self.crop else []) + self.augmentations, image
        #     )
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            # if self.crop:
            #     crop_tfm = utils.gen_crop_transform_with_instance(
            #         self.crop.get_crop_size(image.shape[:2]),
            #         image.shape[:2],
            #         np.random.choice(dataset_dict["annotations"]),
            #     )
            #     image = crop_tfm.apply_image(image)
            # image, transforms = T.apply_augmentations(self.augmentations, image)
            # if self.crop:
            #     transforms = crop_tfm + transforms
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.min_box_side_len,
                self.proposal_topk,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)
                if not self.use_attribute:
                    anno.pop("attribute_ids")

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances_with_attributes(
                annos,
                image_shape,
                mask_format=self.instance_mask_format,
                load_attributes=self.use_attribute,
                max_attr_per_ins=self.max_attr_per_ins,
            )
            # if self.crop and instances.has("gt_masks"):
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # if "sem_seg_file_name" in dataset_dict:
        #     with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
        #         sem_seg_gt = Image.open(f)
        #         sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        #     sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        #     sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        #     dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict


class TestDatasetMapper(DatasetMapper):
    """
    Extend DatasetMapper for feature extraction.
    """

    def __init__(self, cfg, is_train=False):
        super().__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, image)
        except OSError:
            return

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_augmentations(
                ([self.crop] if self.crop else []) + self.augmentations, image
            )
        else:
            if self.crop:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_augmentations(self.augmentations, image)
            if self.crop:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        return dataset_dict
