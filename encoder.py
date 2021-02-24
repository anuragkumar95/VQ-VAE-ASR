import torch.nn as nn
import numpy as np
import os
import sys
from modules import Conv, Dense


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_pre = Conv(layers=2, stride=1, kernel=3, in_dim=in_dim)
        self.conv_strided = Conv(layers=1, stride=2, kernel=4, residual=False)
        self.conv_post = Conv(layers=2, stride=1, kernel=3)
        self.dense = Dense(layers=4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.conv_strided(x)
        x = self.conv_post(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        return self.dense(x)