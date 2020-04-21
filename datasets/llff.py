import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .utils import *

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """


    def normalize(v):
        """Normalize a vector."""
        return v/np.linalg.norm(v)

    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    poses_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return poses_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
    """

    poses_avg = average_poses(poses) # (3, 4)
    poses_avg_homo = np.eye(4)
    poses_avg_homo[:3] = poses_avg # convert to homogeneous coordinate for faster computation
                                   # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(poses_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378)):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*.JPG')))
                      # load full resolution image then resize
        assert len(poses_bounds) == len(image_paths), \
            'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        assert H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        poses = center_poses(poses)

        # Step 3: correct scale so that the near plane is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75 # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val/test
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(image_paths[:-1]): # exclude the last image
                c2w = torch.FloatTensor(poses[i])

                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
                                 # near plane is always at 1.0
                                 # See https://github.com/bmild/nerf/issues/34

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             torch.zeros_like(rays_o[:, :1]),
                                             torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                                 # near and far in NDC are always 0 and 1
                                 # See https://github.com/bmild/nerf/issues/34

            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        else:
            self.c2w_test = poses[-1]
            self.image_path_test = image_paths[-1]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return 1 # only validate/test one image

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            c2w = torch.FloatTensor(self.c2w_test)

            img = Image.open(self.image_path_test)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3)

            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                          self.focal, 1.0, rays_o, rays_d)

            rays = torch.cat([rays_o, rays_d, 
                              torch.zeros_like(rays_o[:, :1]),
                              torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      }

        return sample