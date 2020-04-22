import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        False,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    dataset = dataset_dict[args.dataset_name](args.root_dir, 'test',
                                              img_wh=(w, h))

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    disps = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.chunk,
                                    dataset.white_back)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        img_pred = (img_pred*255).astype(np.uint8)
        
        depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
        depth_pred[depth_pred==0] = np.inf
        disp_pred = 1/depth_pred
        disp_pred = (disp_pred - disp_pred.min())/(disp_pred.max() - disp_pred.min())
        disp_pred = (disp_pred*255).astype(np.uint8)

        imgs += [img_pred]
        disps += [disp_pred]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred)
        imageio.imwrite(os.path.join(dir_name, f'disp_{i:03d}.png'), disp_pred)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_disp.gif'), disps, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')