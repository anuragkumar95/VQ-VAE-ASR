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
    def __init__(self, layers, stride, kernel, in_dim=None, residual=True, transpose=False):
        super().__init__()
        self.cnvs = nn.ModuleList()
        if in_dim:
            self.input_layer = nn.Conv1d(in_channels=in_dim, 
                                         out_channels=128,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=1)
        else:
            layers += 1 
            self.input_layer=None
        for i in range(layers-1):
            if transpose:
                if i == layers-2:
                    out_ch = 39
                else:
                    out_ch = 128
                layer = nn.ConvTranspose1d(in_channels=128, 
                                           out_channels=out_ch,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=1)
            else:
                layer = nn.Conv1d(in_channels=128, 
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
            x = layer(x)
            x = self.relu(x)
        return x
        
