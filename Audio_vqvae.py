import torch
import torch.nn as nn
import torch.nn.functional as F
from vq_vae import VectorQuantizer
from encoder import Encoder
from decoder import Decoder


class audio_vqvae(nn.Module):
    def __init__(self, input_dim, hid_dim, enc_dim, K=512):
        super().__init__()
        self.encoder = Encoder(in_dim=input_dim, hid_dim=hid_dim, enc_dim=enc_dim)
        self.vq = VectorQuantizer(num_embeddings=K, 
                                  embedding_dim=enc_dim, 
                                  commitment_cost=0.25)
        self.decoder = Decoder(in_dim=enc_dim, hid_dim=hid_dim, out_dim=input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        #z = z.permute(0,2,1)
        print("Z:", z.shape)
        vq_loss, quantized, perplexity, _, _, encoding_indices, losses, _, _, _, concatenated_quantized = self.vq(z)
        quantized = quantized.permute(0,2,1)
        print("z':", quantized.shape)
        
        reconstructed = self.decoder(quantized)
        print("reconstructed_x:", reconstructed.shape)
        input_features_size = x.size(2)
        output_features_size = reconstructed.size(2)

        return reconstructed, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized

    