from __future__ import absolute_import, division, print_function

from .encoder import Encoder
from .decoder import Decoder
from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder
from .layers import SSIM, Backproject, Project
import torch
import torch.nn.functional as F
from .resnet import resnet18, resnet34, resnet50, resnet101

import torch.nn as nn
from typing import Dict
class mono_fm_joint(nn.Module):
    def __init__(self, n_channels):
        super(mono_fm_joint, self).__init__()
        self.device = 'cuda'
        depth_num_layers = 18

        self.DepthEncoder = DepthEncoder(depth_num_layers)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)

        self.Encoder = Encoder(depth_num_layers)
        self.Decoder = Decoder(self.Encoder.num_ch_enc)
        # self.ssim = SSIM()
        #
        #
        # self.ssim_weight = self.opt.ssim_weight
        # self.l1_weight = self.opt.l1_weight



    def forward(self, inputs):
        outputs = self.DepthDecoder(self.DepthEncoder(inputs))

        # if self.training:
        #     features = self.Encoder(inputs)
        #     outputs.update(self.Decoder(features, 0))
        #     loss_dict = self.compute_losses(inputs, outputs, features)
        #     return outputs
        return outputs