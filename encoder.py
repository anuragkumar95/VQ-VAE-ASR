import torch.nn as nn
import numpy as np
import os
import sys


class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)

class Conv(nn.Module):
    def __init__(self, layers, stride, kernel, in_dim=None, residual=True):
        super().__init__()
        self.cnvs = nn.ModuleList()
        if in_dim:
            self.input_layer = nn.Conv2d(in_channels=in_dim, 
                                         out_channels=128,
                                         kernel_size=kernel,
                                         stride=stride)
        else:
            self.input_layer=None
        for i in range(layers):
            layer = nn.Conv2d(in_channels=128, 
                              out_channels=128,
                              kernel_size=kernel,
                              stride=stride,
                              padding=1)
            if residual:
                layer = Residual(layer)
            self.cnvs.append(layer)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.input_layer:
            x = self.input_layer(x)
            x = self.relu(x)
        for layer in self.cnvs:
            x = layer(x)
            x = self.relu(x)
        return x

class Dense(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.dense = nn.ModuleList()
        self.relu = nn.ReLU()
        for i in range(layers):
            self.dense.append(Residual(nn.Linear(in_features=128, out_features=128)))
        
    def forward(self, x):
        for layer in self.dense:
            print(x.shape)
            x = layer(x)
            x = self.relu(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_pre = Conv(layers=1, stride=1, kernel=3, in_dim=in_dim)
        self.conv_strided = Conv(layers=1, stride=2, kernel=4, residual=False)
        self.conv_post = Conv(layers=2, stride=1, kernel=3)
        self.dense = Dense(layers=4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.conv_strided(x)
        x = self.conv_post(x)
        x = self.dense(x)
        print("DENSE", x.shape)

        return x