import os.path as osp
from collections import OrderedDict
from itertools import chain
from typing import Tuple, Optional, List, NamedTuple

import cv2
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from rlkit.misc.eval_util import create_stats_ordered_dict
from torch.distributions.kl import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.loss import LossFunction
from rlkit.core.timer import timer
from rlkit.misc import ml_util
from rlkit.torch.distributions import MultivariateDiagonalNormal, Distribution
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_torch_trainer import VAE, compute_vae_terms
from rlkit.visualization.image import combine_images_into_grid

LossStatistics = OrderedDict


class SVAELosses(NamedTuple):
    set_vae_loss: torch.Tensor
    discriminator_loss: torch.Tensor
    set_i: int


def compute_prior(q_z: Distribution):
    if not isinstance(q_z, MultivariateDiagonalNormal):
        raise NotImplementedError()
    second_moment = (q_z.variance + q_z.mean**2).mean(dim=0, keepdim=True)
    first_moment = q_z.mean.mean(dim=0, keepdim=True)
    variance = second_moment - first_moment**2
    stddev = torch.sqrt(variance)
    return MultivariateDiagonalNormal(loc=first_moment, scale_diag=stddev)


