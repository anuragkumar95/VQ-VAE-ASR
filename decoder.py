#https://github.com/swasun/VQ-VAE-Speech/blob/master/src/modules/jitter.py

import sys
import pathlib
sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute()) + '/pytorch-wavenet')

import numpy as np
import torch
import torch.nn as nn
from wavenet_model import WaveNetModel, load_latest_model_from, load_to_cpu
from modules import Conv

class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized

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
    def __init__(self, in_dim=None):
        super().__init__()
        self.jitter = Jitter()
        self.pre_conv = Conv(layers=1, stride=1, kernel=3, in_dim=in_dim)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_mid = Conv(layers=2, stride=1, kernel=3)
        self.conv_post = Conv(layers=3, stride=1, kernel=3, residual=False, transpose=True)

    def forward(self, x):
        x = self.jitter(x)
        x = self.pre_conv(x)
        print("Before upsample:",x.shape)
        x = self.upsample(x)
        print("After upsample:",x.shape)
        x = self.conv_mid(x)
        print("After transpose conv:",x.shape)
        return self.conv_post(x)




if __name__ == '__main__':
    dec = Decoder()
    print("Successfully initialised..")