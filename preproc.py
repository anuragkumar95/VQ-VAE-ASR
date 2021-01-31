import torchaudio
import torch
import torch.utils.data as data
import torch.nn as nn
import os
import numpy as np
import librosa
from tqdm import tqdm
import csv
import pandas as pd
from string import punctuation

def collate_custom(data):
    '''
    For batch
    '''
    sample_rate = 160000
    get_MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                          n_mfcc = 13,
                                          log_mels=True)
    get_deltas = torchaudio.transforms.ComputeDeltas()
    def maxlen_fn(paths, fn):
        max_len=0
        for f in paths:
            sig, sr = torchaudio.load(audio, sr=sample_rate)
            sig = fn(torch.tensor(sig))
            if sig.shape[1] > max_len:
                max_len = sig.shape[1]
        return int(max_len)

    
    batch = []
    batch_mask = []
    transcripts = data["sentence"]
    paths = data["path"]
    maxlen = maxlen_fn(paths, get_MFCC)
    for audio in paths:
        audio, sr = torchaudio.load(audio, sr=sample_rate)
        #Extract features...
        mfcc = get_MFCC(audio)
        deltas = get_MFCC(audio)
        ddeltas = get_deltas(deltas)
        feature = torch.cat([mfcc, deltas, ddeltas], dim=1).squeeze(0)
        #Calculate masks...
        mask = torch.ones(1, feature.shape[1])
        #Calculate padding...
        mask = nn.ZeroPad2d(padding=(0, maxlen-feature.shape[1], 0, 0))(mask)
        batch_audio = nn.ZeroPad2d(padding=(0, maxlen-feature.shape[1], 0, 0))(feature)
        
        batch.append(batch_audio)
        batch_mask.append(mask)

    return {"feature":torch.stack(batch_audio), 
            "mask":torch.stack(batch_mask), 
            "transcript":transcripts}

class Data(data.Dataset):
    def __init__(self, csv_path, data_path):
        self.csv = pd.read_csv(csv_path, sep='\t')
        self.path = [os.path.join(data_path, fpath) for fpath in self.csv['path']]
        self.transcripts = [''.join([char.lower() for char in sent if char not in punctuation])
                            for sent in self.csv['sentence']]

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {"path":self.path[idx],
                  "transcript":self.transcripts[idx]}
        return sample

class DataVAE(data.Dataset):
    def __init__(self, csv_path, data_path):
        self.csv = pd.read_csv(csv_path, sep='\t')
        self.path = [os.path.join(data_path, fpath) for fpath in self.csv['path']]

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.path[idx]


def collate_vae(data):
    '''
    For batch
    '''
    sample_rate = 160000
    get_MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                          n_mfcc=13,
                                          log_mels=True)
    get_deltas = torchaudio.transforms.ComputeDeltas()

    def maxlen_fn(paths, fn):
        max_len=0
        for f in paths:
            sig, sr = torchaudio.load(f)
            sig = fn(torch.tensor(sig))
            if sig.shape[1] > max_len:
                max_len = sig.shape[1]
        return int(max_len)

    
    batch = []
    maxlen = maxlen_fn(data, get_MFCC)

    for audio in data:
        audio, sr = torchaudio.load(audio)
        #Extract features...
        mfcc = get_MFCC(audio)
        deltas = get_deltas(mfcc)
        ddeltas = get_deltas(deltas)
        print(mfcc.shape, deltas.shape, ddeltas.shape)
        feature = torch.cat([mfcc, deltas, ddeltas], dim=1).squeeze(0)
        batch_audio = nn.ZeroPad2d(padding=(0, maxlen-feature.shape[1], 0, 0))(feature)
        print('sample:', batch_audio.shape)
        batch.append(batch_audio)

    return torch.stack(batch)