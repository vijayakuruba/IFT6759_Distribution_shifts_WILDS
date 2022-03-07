import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x_center = _sample_uniform(0, w)
    y_center = _sample_uniform(0, h)

    x0 = int(max(0, x_center - v / 2.0))
    y0 = int(max(0, y_center - v / 2.0))
    x1 = min(w, x0 + v) 
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def _sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()


class CutoutAugment:
    #def __init__(self):
        #assert n >= 1, "RandAugment N has to be a value greater than or equal to 1."
        #self.n = n
        #self.augmentation_pool = augmentation_pool

    def __call__(self, img):
        #ops = [
        #    self.augmentation_pool[torch.randint(len(self.augmentation_pool), (1,))]
        #    for _ in range(self.n)
        #]
        #for op, min_val, max_val in ops:
        #    val = min_val + float(max_val - min_val) * _sample_uniform(0, 1)
        #    img = op(img, val)
        cutout_val = _sample_uniform(0, 1) * 0.5
        img = Cutout(img, cutout_val)
        return img
        return img