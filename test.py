import torch
from vq_modules import VectorQuantizedVAE
from decoder import Decoder
from encoder import Encoder

x = torch.rand(10, 39, 1200)
#model = VectorQuantizedVAE(input_dim = 39, dim=128)
#model = Encoder(in_dim=1)
#vq = VectorQuantizedVAE(input_dim = 39, dim=128)
#dec = Decoder(in_dim=128)
#_, e_out, vq_out = vq(x)
#vq_out = vq_out.squeeze(1).permute(0,2,1).contiguous()
#x_tilde, z_e_x, z_q_x = model(x)
#print("Input:",x.shape,"VQ out:",z_q_x.shape,"Output:",x_tilde.shape)
e = Encoder(in_dim=39, hid_dim=768)
o = e(x)