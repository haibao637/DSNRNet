import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import cv2

import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))
def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)

