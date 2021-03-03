import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

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
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.N_importance > 0:
        nerf_fine = NeRF()
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    imgs, depth_maps, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(f'depth_{i:03d}', 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.gif'), depth_imgs_, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
