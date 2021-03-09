import torch.nn as nn
import torch
import numpy as np
import os
import sys
from modules import Conv, Dense
from preproc import collate_vae, DataVAE


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, enc_dim):
        super().__init__()
        self.conv_pre = Conv(layers=2, stride=1, kernel=3, hid_dim = hid_dim, in_dim=in_dim)
        self.conv_strided = Conv(layers=1, stride=2, kernel=4, hid_dim = hid_dim, residual=False)
        self.conv_post = Conv(layers=2, stride=1, kernel=3, hid_dim = hid_dim)
        self.dense = Dense(layers=4, hid_dim=hid_dim)
        self.out = nn.Linear(in_features=hid_dim, out_features=enc_dim)

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.conv_strided(x)
        x = self.conv_post(x)
        x = x.permute(0,2,1)
        x = self.dense(x)
        x = self.out(x)
        out = x.permute(0, 2, 1)
        print("output shape:", out.shape)
        return x


train_dataset = DataVAE('/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/train.tsv',
                         '/nobackup/anakuzne/data/cv/cv-corpus-5.1-2020-06-22/eu/clips/')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=10, 
                                          shuffle=False, 
                                          pin_memory=True, 
                                          collate_fn=collate_vae)

device = torch.device("cuda:2")
model = Encoder(39, 768, 128)
model.cuda()
model = model.to(device)
model = nn.DataParallel(model, device_ids=[2, 3])

for batch in train_loader:
    out = model(batch)
    print(out.shape)