#!/bin/python
import numpy as np

import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, stride=1, padding='same'):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

class skip_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = block(3,64,5,1)
        self.block2 = block(64,64,5,1)
        self.block3 = block(64,128,3,1)
        self.block4 = block(128,128,3,1)
        self.block5 = block(128,128,3,1)
        self.block6 = block(128,128,3,1)
        self.block7 = block(128,64,3,1)
        self.block8 = block(64,64,3,1)
        self.block9 = block(64,3,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #1
        x = self.block1(x)
        x1 = x.clone()
        x = self.pool(x)
        #2
        x = self.block2(x)
        x2 = x.clone()
        x = self.pool(x)
        #3
        x = self.block3(x)
        x3 = x.clone()
        x = self.pool(x)
        #4
        x = self.block4(x)
        x = self.pool(x)
        #5
        x = self.block5(x)
        x = self.up(x)
        #6
        x = self.block6(x)
        x = self.up(x)
        x = x + x3
        #7
        x = self.block7(x)
        x = self.up(x)
        x = x + x2
        #8
        x = self.block8(x)
        x = self.up(x)
        x = x + x1
        #9
        out = self.block9(x)
        return out