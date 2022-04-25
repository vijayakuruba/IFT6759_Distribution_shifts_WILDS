import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))



WEAK_AUGMENTATION_POOL = [
    (TranslateX, -0.3, 0.3),
    (TranslateY, -0.3, 0.3),
]


def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()


class WeakAugment:
    def __init__(self, augmentation_pool):
        self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        ops = [
            self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
        ]
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
            img = op(img, val)
        return img
