import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .utils import *

class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2
        self.far = 6

        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                c2w = torch.FloatTensor(frame['transform_matrix'])

                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh)
                img = self.transform(img) # (4, H, W)
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3] # (H*W, 3) RGB
                # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # (H*W, 3) composite alpha to RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.img_wh[1], self.img_wh[0],
                                          self.focal, c2w)
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0], self.focal,
                                              1.0, rays_o, rays_d)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (H*W, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (H*W*len(self.meta['frames]), 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (H*W*len(self.meta['frames]), 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return 1 # only validate one image each epoch

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = np.random.choice(self.meta['frames'], 1)[0] # randomly sample an image
            c2w = torch.FloatTensor(frame['transform_matrix'])

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh)
            img = self.transform(img) # (4, H, W)
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3] # (H*W, 3) RGB
            # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # (H*W, 3) composite alpha to RGB

            rays_o, rays_d = get_rays(self.img_wh[1], self.img_wh[0],
                                      self.focal, c2w)
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0], self.focal,
                                          1.0, rays_o, rays_d)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img}

        return sample