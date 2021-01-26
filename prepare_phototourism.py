import argparse
from datasets import PhototourismDataset
import numpy as np
import os

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()
    print('Preparing cache ...')
    dataset = PhototourismDataset(args.root_dir, 'train', args.img_downscale)
    np.save(os.path.join(args.root_dir, f'cache_rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cache_rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())
    print('Data cache saved!')