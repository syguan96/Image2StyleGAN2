"""
StyleGAN2 Inversion, refer to Image2StyleGAN
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from stylegan2.model import Generator
from lpips.loss import LPIPS

parser = argparse.ArgumentParser('Invertor')
# training params
parser.add_argument('--expname', type=str, default='exp1', help='experiment name')
parser.add_argument('--expdir', type=str, default='exps', help='dirs of experiments')
parser.add_argument('--imagename', type=str, required=True, help='input image name')
parser.add_argument('--stylegan2_path', type=str, default='data/stylegan2-ffhq-config-f.pt', help='path of pretrianed stylegan model')
parser.add_argument('--iter_num', type=int, default=1000, help='iteration steps')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')


class Invertor():
    def __init__(self,options):
        self.options = options
        self.device = torch.device('cuda')
        self.exppath = os.path.join(self.options.expdir, self.options.expname)
        os.makedirs(self.exppath, exist_ok=True)
        self.logger = SummaryWriter(log_dir=self.exppath)

        # load stylegan2
        self.load_stylegan2_G()

        # setup image transform
        self.image_transforms = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        # setup criterion
        self.lpips_criterion = LPIPS(net_type='alex').to(self.device).eval()
        self.MSE_criterion = nn.MSELoss().to(self.device)


    def load_stylegan2_G(self,):
        """load stylegan2 generator"""
        ckpt = torch.load(self.options.stylegan2_path)
        self.G = Generator(1024, style_dim=512, n_mlp=8)
        self.G.load_state_dict(ckpt['g_ema'], strict=True)
        self.G.eval()
        # suppose we only use one GPU card
        self.G.to(self.device)
        self.mean_latent = ckpt['latent_avg'].to(self.device).repeat(1,18,1)


    def write_summaries(self, results, step):
        for k in results.items():
            if 'loss' in k:
                self.logger.add_scalar(f'{k}', results[k], step)
            elif 'image' in k:
                self.logger.add_images(f'{k}', results[k], step)
        return


    def read_image(self,):
        image = Image.open(self.options.imagename)
        return self.image_transforms(image).to(self.device).unsqueeze(0)
    

    def tensor2numpy(self, images):
        """ we assume the shape of image is (1, C, H, W), and it's a cuda pytorch tensor
        """
        images = torch.clamp(images.detach(), min=-1, max=1)
        images = ((images+1)/2)*255
        images = images.permute(0,2,3,1).detach().cpu().numpy().astype('uint8')
        return images


    def initial_latentcode(self, latent_type):
        if latent_type == 'randn':
            return torch.randn((1,18,512)).to(self.device)
        elif latent_type == 'zero':
            return torch.zeros((1,18,512)).to(self.device)
        elif latent_type == 'mean':
            return torch.from_numpy(np.load('data/mean_latent.npy')).float().to(self.device).unsqueeze(0)
        elif latent_type == 'mean_ckpt':
            return self.mean_latent
        else:
            raise NotImplementedError


    def run(self,):
        image = self.read_image()
        latentcode = self.initial_latentcode(latent_type='mean_ckpt')
        latentcode.requires_grad = True
        optimizer = torch.optim.Adam([latentcode], lr=self.options.learning_rate)
        for step in tqdm(range(self.options.iter_num)):
            decoded_image, _ = self.G([latentcode], input_is_latent=True, randomize_noise=False, return_latents=True)
            decoded_image = F.interpolate(decoded_image, size=(256, 256), mode='bicubic')
            lpipsloss = self.lpips_criterion(decoded_image, image)
            mseloss = self.MSE_criterion(decoded_image, image)

            loss = lpipsloss + mseloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dataitems = {'lpipsloss': lpipsloss, 
                         'mseloss': mseloss}
            if step % 50 == 0 or step == self.options.iter_num:
                self.write_summaries(dataitems, step)
                decoded_image_np = self.tensor2numpy(decoded_image)
                decoded_image_np = Image.fromarray(decoded_image_np[0])
                decoded_image_np.save(f'{self.exppath}/{step}.png')
        print('Finished')
            

if __name__ == '__main__':
    options = parser.parse_args()
    invertor = Invertor(options)
    invertor.run()

