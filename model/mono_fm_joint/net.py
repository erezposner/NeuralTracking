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
import torch.nn.functional as F

class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, padding=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Disparity_Refinement_Network(nn.Module):
    def __init__(self):
        super(Disparity_Refinement_Network, self).__init__()
        self.conv1_dis = nn.Conv2d(1, 16, kernel_size=3, padding=1, dilation=1)

        self.bn1_dis = nn.BatchNorm2d(16)
        self.act1_dis = nn.LeakyReLU(negative_slope=0.2)

        self.conv1_ir = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_ir = nn.BatchNorm2d(16)
        self.act1_ir = nn.LeakyReLU(negative_slope=0.2)

        self.res_block_1_dis = BasicResBlock(16, 16, stride=1)
        self.res_block_2_dis = BasicResBlock(16, 16, stride=1, dilation=2, padding=2)

        self.res_block_1_ir = BasicResBlock(16, 16, stride=1)
        self.res_block_2_ir = BasicResBlock(16, 16, stride=1, dilation=2, padding=2)

        self.res_block_3 = BasicResBlock(32, 32, stride=1, dilation=4, padding=4)
        self.res_block_4 = BasicResBlock(32, 32, stride=1, dilation=8, padding=8)
        self.res_block_5 = BasicResBlock(32, 32, stride=1, dilation=1, padding=1)
        self.res_block_6 = BasicResBlock(32, 32, stride=1, dilation=1, padding=1)
        self.pred = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, dis, ir):
        x_dis = self.conv1_dis(dis)
        x_dis = self.bn1_dis(x_dis)
        x_dis = self.act1_dis(x_dis)
        x_dis = self.res_block_1_dis(x_dis)
        x_dis = self.res_block_2_dis(x_dis)

        x_ir = self.conv1_ir(ir)
        x_ir = self.bn1_ir(x_ir)
        x_ir = self.act1_dis(x_ir)
        x_ir = self.res_block_1_ir(x_ir)
        x_ir = self.res_block_2_ir(x_ir)

        x = torch.cat((x_dis, x_ir), 1)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.res_block_6(x)
        y = self.pred(x)
        y = torch.tanh(y)
        return y
class mono_fm_joint(nn.Module):
    def __init__(self, n_channels):
        super(mono_fm_joint, self).__init__()
        self.device = 'cuda'
        depth_num_layers = 18

        self.DepthEncoder = DepthEncoder(depth_num_layers)
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)
        self.refine_net = Disparity_Refinement_Network()

        self.Encoder = Encoder(depth_num_layers)
        self.Decoder = Decoder(self.Encoder.num_ch_enc)
        # import os
        # os.environ['TORCH_MODEL_ZOO'] = 'model/mono_fm_joint/.'
        # os.environ['TORCH_HOME'] = 'model/mono_fm_joint/.'
        from torchvision import models
        # self.fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().cuda() #TODO

        # self.ssim = SSIM()
        #
        #
        # self.ssim_weight = self.opt.ssim_weight
        # self.l1_weight = self.opt.l1_weight

    # segment(inputs)
    # Define the helper function
    def segment(self, inputs):
        import torchvision.transforms as T

        import numpy as np
        def decode_segmap(image, nc=21):
            label_colors = np.array([(0, 0, 0),  # 0=background
                                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                     (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                     # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                     (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                     # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                     (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

            r = np.zeros_like(image).astype(np.uint8)
            g = np.zeros_like(image).astype(np.uint8)
            b = np.zeros_like(image).astype(np.uint8)

            for l in range(0, nc):
                idx = image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]

            rgb = np.stack([r, g, b], axis=2)
            return rgb



        trf = T.Compose([T.Resize(256),
                         # T.CenterCrop(224),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        inp = trf(inputs)

        out = self.fcn(inp)['out']
        om = torch.argmax(out.squeeze(), dim=1).detach().cpu().numpy()

        # import matplotlib.pyplot as plt
        # rgb = decode_segmap(om[0])
        # plt.imshow(rgb);
        # plt.show()
        ttt = torch.argmax(out, dim=1).unsqueeze(1).float()
        upsampled = F.interpolate(ttt, [inputs.shape[2], inputs.shape[3]], mode='nearest')
        trf = T.Compose([
                         T.Normalize(mean=[0.485],
                                     std=[0.229])])
        # plt.imshow(upsampled[1][0].detach().cpu().numpy());
        # plt.show()
        return trf(upsampled / 21)

    def forward(self, inputs):
        # with torch.no_grad():
        #
        #     segs = self.segment(inputs)
        # seg_img = torch.cat((inputs, segs),dim=1)
        outputs = self.DepthDecoder(self.DepthEncoder(inputs))
        # outputs = self.DepthDecoder(self.DepthEncoder(seg_img))
        depth_layer_to_refine = outputs[('depth', 0, 0)]
        upsampled_depth_layer_to_refine = F.interpolate(depth_layer_to_refine, [inputs.shape[2], inputs.shape[3]], mode="nearest")
        # refined_depth = self.refine_net(upsampled_depth_layer_to_refine, inputs)
        outputs[('depth', -1, -1)] = upsampled_depth_layer_to_refine #+ refined_depth #TODO check
        # if self.training:
        #     features = self.Encoder(inputs)
        #     outputs.update(self.Decoder(features, 0))
        #     loss_dict = self.compute_losses(inputs, outputs, features)
        #     return outputs
        return outputs