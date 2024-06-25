
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

class FGEB(nn.Module):
    def __init__(self, output_chl, reduction=8):
        super(FGEB, self).__init__()
        # self.register_buffer("precomputed_dct_weights", get_dct_weights(...))
        self.output_chl = output_chl
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv1_1 = nn.Conv2d(
        #     in_channels=self.output_chl,
        #     out_channels=self.output_chl,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )
        # self.concat_project = nn.Sequential(
        #     nn.Conv2d(self.output_chl + 1, 1, 1, 1, 0, bias=False),
        #     nn.ReLU()
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.output_chl * 3, self.output_chl // reduction),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.output_chl // reduction, self.output_chl),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, y,z):
        b, c, _, _ = x.size()
        out_x_avg = self.avg_pool(x).view(b, c)
        out_y_avg = self.avg_pool(y).view(b, c)
        out_z_avg = self.avg_pool(z).view(b, c)
        # out_x_max = self.max_pool(x).view(b, c)
        # out_y_max = self.max_pool(y).view(b, c)
        # out_x = out_x_avg + out_x_max
        # out_y = out_y_avg + out_y_max
        # # (b,c,N,1)
        # out_x2 = self.conv1_1(out_x)
        # # (b,c,1,N)
        # out_y2 = self.conv1_1(out_y)

        # h = out_x2.size(2)
        # w = out_y2.size(3)
        # f = torch.zeros_like(out_x2).cuda()
        # concat_feature_avg = self.fc(torch.cat([out_x_avg, out_y_avg], dim=1)).view(b,c,1,1)
        # concat_feature_max = self.fc(torch.cat([out_x_max, out_y_max], dim=1)).view(b,c,1,1)
        # concat_feature = concat_feature_avg + concat_feature_max
        concat_feature = torch.cat([out_x_avg, out_y_avg,out_z_avg], dim=1)
        f = self.fc(concat_feature).view(b, c, 1, 1)
        # f = self.fc(concat_feature).view(b, c, 1, 1)
        # return y * self.sigmoid(concat_feature)
        return z * f

