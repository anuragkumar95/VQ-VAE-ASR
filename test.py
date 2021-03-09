import torch
from vq_modules import VectorQuantizedVAE
from decoder import Decoder
from encoder import Encoder
from Audio_vqvae import audio_vqvae

x = torch.rand(10, 39, 1200)
model = audio_vqvae(input_dim=39, hid_dim=768, enc_dim=64, K=512)

reconstructed, vq_loss, losses, perplexity, \
    encoding_indices, concatenated_quantized = model(x)
print(reconstructed.shape, vq_loss)

