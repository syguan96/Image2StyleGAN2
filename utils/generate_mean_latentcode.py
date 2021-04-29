"""
Generate mean latent code
"""
import sys
sys.path.append('.')
import os
import numpy as np
import torch

from stylegan2.model import Generator

Num_latents = 10000

G = Generator(1024, 512, 8).cuda()
G.load_state_dict(torch.load('data/stylegan2-ffhq-config-f.pt')['g_ema'], strict=False)

G.eval()
with torch.no_grad():
    noises = torch.randn(Num_latents, 512).cuda()
    latents = G.style(noises)
    mean_latent = latents.mean(0)
    np.save('data/mean_latent', mean_latent.cpu().numpy())