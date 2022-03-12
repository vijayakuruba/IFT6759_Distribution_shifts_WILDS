import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE
from utils import cross_entropy_with_logits_loss

import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'data_augmentation')
sys.path.append(path)

from mixcut_augment.mixcutaugment import CutMixCriterion

def initialize_loss(loss, config):

    lfn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    eval_lfn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    if loss == 'cross_entropy':
        loss_fn = ElementwiseLoss(loss_fn=lfn)#nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        eval_loss_fn = ElementwiseLoss(loss_fn=eval_lfn)#nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        if config.mixcut:
            loss_fn = CutMixCriterion(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))


    elif loss == 'lm_cross_entropy':
        loss_fn =  MultiTaskLoss(loss_fn=lfn)#nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        eval_loss_fn =  MultiTaskLoss(loss_fn=eval_lfn)#nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        if config.mixcut:
            loss_fn = CutMixCriterion(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        loss_fn =  MSE(name='loss')
        eval_loss_fn =  MSE(name='loss')

    elif loss == 'multitask_bce':
        loss_fn =  MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))
        eval_loss_fn =  MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))


    elif loss == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        loss_fn =  ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))
        eval_loss_fn =  ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))
        if config.mixcut:
            loss_fn = CutMixCriterion(loss_fn=FasterRCNNLoss(config.device))


    elif loss == 'cross_entropy_logits':
        loss_fn =  ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)
        eval_loss_fn =  ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)
        if config.mixcut:
            loss_fn = CutMixCriterion(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')
        #loss_fn = ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    return {'loss': loss_fn, 'eval_loss': eval_loss_fn}
