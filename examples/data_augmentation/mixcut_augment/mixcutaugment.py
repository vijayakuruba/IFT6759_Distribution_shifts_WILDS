import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from wilds.common.metrics.loss import ElementwiseLoss



def cutmix(batch, alpha):
    data, targets, metadata = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    lam = lam * torch.ones(targets.shape)

    targets = torch.stack(list((targets, shuffled_targets, lam)), dim=0)

    return data, targets, metadata

class MixcutCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion(ElementwiseLoss):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(loss_fn=loss_fn, name=name)

    def _compute_element_wise(self, preds, targets):
        targets1, targets2, lam = targets.chunk(3)
        lam =lam.data[0]
        targets1 = torch.squeeze(targets1, 0).type(torch.LongTensor).to(targets1.device)
        targets2 = torch.squeeze(targets2, 0).type(torch.LongTensor).to(targets1.device)

        return lam * self.loss_fn(
            preds, targets1) + (1 - lam) * self.loss_fn(preds, targets2)
