import copy
from typing import List

import numpy as np
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment
from data_augmentation.cutout_augment.cutoutaugment import CutoutAugment
from data_augmentation.weak_augment.weakaugment import WEAK_AUGMENTATION_POOL, WeakAugment

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform_name=None
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.
    """
    if transform_name is None:
        return None

    # For images
    normalize = True
    if transform_name == "image_base":
        transform_steps = get_image_base_transform_steps(config, dataset)
    elif transform_name == "image_resize":
        transform_steps = get_image_resize_transform_steps(
            config, dataset
        )
    elif transform_name == "image_resize_and_center_crop":
        transform_steps = get_image_resize_and_center_crop_transform_steps(
            config, dataset
        )

    else:
        raise ValueError(f"{transform_name} not recognized")

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    if additional_transform_name == "fixmatch":
        transformations = add_fixmatch_transform(
                config, dataset, transform_steps, default_normalization
            )
        transform = MultipleTransforms(transformations)
    elif additional_transform_name == "randaugment":
        transform = add_rand_augment_transform(
                config, dataset, transform_steps, default_normalization
            )
    elif additional_transform_name == "weak":
        transform = add_weak_transform(
            config, dataset, transform_steps, normalize, default_normalization
        )
    elif additional_transform_name == "cutout":
        transform = add_cutout_transform(
            config, dataset, transform_steps, normalize, default_normalization
        )
    elif additional_transform_name == "weak+":
        transform = add_weakenhanced_transform(
            config, dataset, transform_steps, default_normalization
        )    
    else:
        if transform_name != "poverty":
            # The poverty data is already a tensor at this point
            transform_steps.append(transforms.ToTensor())
        if normalize:
            transform_steps.append(default_normalization)
        transform = transforms.Compose(transform_steps)

    return transform



def get_image_base_transform_steps(config, dataset) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config.target_resolution is not None:
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_resize_transform_steps(config, dataset)
    target_resolution = _get_target_resolution(config, dataset)
    transform_steps.append(
        transforms.CenterCrop(target_resolution),
    )
    return transform_steps


def get_image_resize_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    return [
        transforms.Resize(scaled_resolution)
    ]

def add_fixmatch_transform(config, dataset, base_transform_steps, normalization):
    return (
        add_weak_transform(config, dataset, base_transform_steps, True, normalization),
        add_rand_augment_transform(config, dataset, base_transform_steps, normalization)
    )


def add_weak_transform(config, dataset, base_transform_steps, should_normalize, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution,
            ),
        ]
    )
    if should_normalize:
        weak_transform_steps.append(transforms.ToTensor())
        weak_transform_steps.append(normalization)
    return transforms.Compose(weak_transform_steps)
    
def add_weakenhanced_transform(config, dataset, base_transform_steps, normalization):
  
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            WeakAugment(
                
                augmentation_pool=WEAK_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(weak_transform_steps)

def add_rand_augment_transform(config, dataset, base_transform_steps, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return transforms.Compose(strong_transform_steps)

def add_cutout_transform(config, dataset, base_transform_steps, should_normalize, normalization):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    cutout_transform_steps = copy.deepcopy(base_transform_steps)
    cutout_transform_steps.extend(
        [
            transforms.RandomCrop(
                size=target_resolution
            ),
            CutoutAugment(),
        ]
    )
    if should_normalize:
        cutout_transform_steps.append(transforms.ToTensor())
        cutout_transform_steps.append(normalization)
    return transforms.Compose(cutout_transform_steps)



def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)
