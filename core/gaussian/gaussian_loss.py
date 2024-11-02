'''
gaussian_loss.py is used to compute the loss of gs
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import imageio
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
from random import randint
from core.gaussian.utils.loss_utils import l1_loss, l2_loss, ssim
from core.gaussian.gaussian_renderer import render
from core.gaussian.utils.graphics_utils import focal2fov, fov2focal
from typing import NamedTuple
from PIL import Image
from smpl.smpl_numpy import SMPL
from smpl.smpl_torch_batch import SMPLModel

class GS3DLoss(nn.Module):
    def __init__(self,
                 ):

        super(GS3DLoss, self).__init__()

    def forward(self, iteration):
        pass