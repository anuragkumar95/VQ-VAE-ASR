import torch.nn as nn
import torch
import numpy as np
import os
import sys
from preproc import DataVAE, collate_vae

#https://github.com/swasun/VQ-VAE-Speech/blob/master/src/modules/jitter.py


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

class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        print("Do residual")
        return x + self.layer(x)

class Conv(nn.Module):
    def __init__(self, layers, stride, kernel, hid_dim, 
                 in_dim=None, residual=True):
        super().__init__()
        self.cnvs = nn.ModuleList()
        if in_dim:
            self.input_layer = nn.Conv1d(in_channels=in_dim, 
                                         out_channels=hid_dim,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=1)
        for i in range(layers-1):
            layer = nn.Conv1d(in_channels=hid_dim, 
                            out_channels=hid_dim,
                            kernel_size=kernel,
                            stride=stride,
                            padding=1)
            if residual:
                layer = Residual(layer)
            self.cnvs.append(layer)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        print(x.shape)
        x = self.input_layer(x)
        x = self.relu(x)
        print("After Input:", x.shape)
        print("len others:", len(self.cnvs))
        for layer in self.cnvs:
            x = layer(x)
            x = self.relu(x)
        return x

class Dense(nn.Module):
    def __init__(self, layers, hid_dim):
        super().__init__()
        self.dense = nn.ModuleList()
        self.relu = nn.ReLU()
        for i in range(layers):
            self.dense.append(Residual(nn.Linear(in_features=hid_dim, out_features=hid_dim)))

    def forward(self, x):
        for layer in self.dense:
            x = layer(x)
            x = self.relu(x)
        return x

#######################################
        
train_dataset = DataVAE('/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/train.tsv',
                         '/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/clips/')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=10, 
                                          shuffle=False, 
                                          pin_memory=True, 
                                          collate_fn=collate_vae)

device = torch.device("cuda:2")
model = Conv(layers=2, stride=1, kernel=3, hid_dim=768, in_dim=39)
model.cuda()
model = model.to(device)
model = nn.DataParallel(model, device_ids=[2, 3])

for batch in train_loader:
    out = model(batch)