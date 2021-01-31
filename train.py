from preproc import Data, collate_custom


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F




dataset = Data('/nobackup/anakuzne/data/cv/target-segments/eu/train.tsv',
                '/nobackup/anakuzne/data/cv/target-segments/eu/')
loader = data.DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_custom)

print(len(loader))