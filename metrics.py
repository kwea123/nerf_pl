import torch
from kornia.losses import ssim as dssim

def mse(image_pred, image_gt):
    return torch.mean((image_pred-image_gt)**2)

def psnr(image_pred, image_gt):
    return -10*torch.log10(mse(image_pred, image_gt))

def ssim(image_pred, image_gt):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, 'mean') # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]