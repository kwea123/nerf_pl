import torch

def mse(image_pred, image_gt):
    return torch.mean((image_pred-image_gt)**2)

def psnr(image_pred, image_gt):
    return -10*torch.log10(mse(image_pred, image_gt))