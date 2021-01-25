import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


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

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered


class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_scale >= 1!'
        self.img_downscale = img_downscale
        self.val_num = max(1, val_num) # at least 1
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        self.files = pd.read_csv(glob.glob(os.path.join(self.root_dir, '*.tsv'))[0], sep='\t')

        # Step 1. load image paths
        imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
        img_path_to_id = {}
        for v in imdata.values():
            img_path_to_id[v.name] = v.id
        img_ids = []
        self.image_paths = {} # {id: filename}
        for filename in list(self.files['filename']):
            id_ = img_path_to_id[filename]
            self.image_paths[id_] = filename
            img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        self.Ks = {} # {id: K}
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
        for id_ in img_ids:
            K = np.zeros((3, 3), dtype=np.float32)
            cam = camdata[id_]
            K[0, 0] = cam.params[0]/self.img_downscale # fx
            K[1, 1] = cam.params[1]/self.img_downscale # fy
            K[0, 2] = cam.params[2]/self.img_downscale # cx
            K[1, 2] = cam.params[3]/self.img_downscale # cy
            K[2, 2] = 1
            self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for id_ in img_ids:
            im = imdata[id_]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.poses = center_poses(poses)

        # Step 4: correct scale
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
        xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
        xyz_world_h = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], -1)
        # Compute near and far bounds for each image individually
        self.nears, self.fars = {}, {} # {id_: distance}
        for i, id_ in enumerate(img_ids):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
            self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
            self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

        max_far = np.fromiter(self.fars.values(), np.float32).max()
        scale_factor = max_far/5 # so that the max far is scaled to 5
        self.poses[..., 3] /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(img_ids)}
        for k, v in self.nears.items():
            self.nears[k] /= scale_factor
        for k, v in self.fars.items():
            self.fars[k] /= scale_factor
            
        # split the img_ids
        img_paths_train = list(self.files[self.files['split']=='train']['filename'])
        self.img_ids_train = [img_path_to_id[path] for path in img_paths_train]
        self.N_images_train = len(self.img_ids_train)
        img_paths_test = list(self.files[self.files['split']=='test']['filename'])
        self.img_ids_test = [img_path_to_id[path] for path in img_paths_test]
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            for id_ in self.img_ids_train:
                c2w = torch.FloatTensor(self.poses_dict[id_])

                img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                              self.image_paths[id_])).convert('RGB')
                img_w, img_h = img.size
                if self.img_downscale > 1:
                    img_w = img_w//self.img_downscale
                    img_h = img_h//self.img_downscale
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                rays_o, rays_d = get_rays(directions, c2w)
                rays_t = id_ * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                             self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val': # use the first image as val image (also in train)
            self.val_id = img_ids[0]

        else: # for testing, create a parametric rendering path
            raise NotImplementedError

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        return self.N_images_test

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}

        elif self.split in ['val', 'test_train']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            else:
                id_ = self.img_ids_train[idx]
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        else:
            raise NotImplementedError

        return sample
