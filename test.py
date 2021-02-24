import torch
from vq_modules import VectorQuantizedVAE
from decoder import Decoder

x = torch.arange(1, 25601, dtype=torch.float32).view(1,1,256,-1)
#model = VectorQuantizedVAE(input_dim = 1, dim=10, K=10)
model = Decoder(in_dim=100)
out = model(x)
print(out.shape)