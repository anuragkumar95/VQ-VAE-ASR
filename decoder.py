import sys
import pathlib
sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute()) + '/pytorch-wavenet')

import numpy as np
import torch
import torch.nn as nn
from wavenet_model import WaveNetModel, load_latest_model_from, load_to_cpu

class Decoder(nn.Module):
    def __init__(self, in_dim=None):
        super().__init__()
        self.model = WaveNetModel(classes=input_dim)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    dec = Decoder()
    print("Successfully initialised..")