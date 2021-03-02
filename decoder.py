
import sys
import pathlib
sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute()) + '/pytorch-wavenet')

import numpy as np
import torch
import torch.nn as nn
#from wavenet_model import WaveNetModel, load_latest_model_from, load_to_cpu
from modules import Conv, Jitter

'''
class Decoder(nn.Module):
    def __init__(self, in_dim=None):
        super().__init__()
        self.jitter = Jitter()
        self.upsample = nn.Upsample(scale_factor=(1, 320))
        self.model = WaveNetModel()

    def forward(self, x):
        x = self.jitter(x)
        x = self.upsample(x)
        return self.model(x.squeeze(1))
'''

class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.jitter = Jitter()
        self.pre_conv = Conv(layers=1, stride=1, kernel=3, hid_dim = hid_dim, in_dim=in_dim, residual=False)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_mid = Conv(layers=4, stride=1, kernel=3, hid_dim = hid_dim)
        self.conv_trans1 = Conv(layers=2, stride=1, kernel=3, hid_dim = hid_dim, out_dim=hid_dim, residual=False, transpose=True)
        self.conv_trans2 = Conv(layers=1, stride=1, kernel=2, hid_dim = hid_dim, out_dim=out_dim, residual=False, transpose=True)

    def forward(self, x):
        print("Before jitter:", x.shape)
        x = self.jitter(x)
        print("After jitter:", x.shape)
        x = self.pre_conv(x)
        print("after pre_conv:", x.shape)
        x = self.upsample(x)
        print("After upsample:", x.shape)
        x = self.conv_mid(x)
        print("After conv_mid", x.shape)
        x = self.conv_trans1(x)
        print("After trans1:", x.shape)
        out = self.conv_trans2(x)
        print("After trans2:", out.shape)
        return out

if __name__ == '__main__':
    dec = Decoder()
    print("Successfully initialised..")