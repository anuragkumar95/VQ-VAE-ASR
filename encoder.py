import torch.nn as nn
import numpy as np
import os
import sys
from modules import Conv, Dense


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.conv_pre = Conv(layers=2, stride=1, kernel=3, hid_dim = hid_dim, in_dim=in_dim)
        self.conv_strided = Conv(layers=1, stride=2, kernel=4, hid_dim = hid_dim, residual=False)
        self.conv_post = Conv(layers=2, stride=1, kernel=3, hid_dim = hid_dim)
        self.dense = Dense(layers=4, hid_dim=hid_dim)
        self.out = nn.Linear(in_features=hid_dim, out_features=64)

    def forward(self, x):
        x = self.conv_pre(x)
        print("Before stride:",x.shape)
        x = self.conv_strided(x)
        print("After stride:",x.shape)
        x = self.conv_post(x)
        print("Conv after stride:", x.shape)
        x = x.permute(0,2,1)
        x = self.dense(x)
        x = self.out(x)
        out = x.permute(0, 2, 1)
        print("output shape:", out.shape)
        return out