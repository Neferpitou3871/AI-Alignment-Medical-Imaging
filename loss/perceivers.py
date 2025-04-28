

import lpips
import torch

import lpips
import torch
import os

class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    def __init__(self, linear_calibration=False, normalize=False, model_dir=None):
        super().__init__()
        
        if model_dir:
            os.environ['TORCH_HOME'] = model_dir  # Set the custom model directory

        self.loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration)
        self.normalize = normalize  # If true, normalize [0, 1] to [-1, 1]

    def forward(self, pred, target):
        if pred.ndim == 5:  # 3D Image: Just use 2D model and compute average over slices 
            depth = pred.shape[2] 
            losses = torch.stack([self.loss_fn(pred[:, :, d], target[:, :, d], normalize=self.normalize) for d in range(depth)], dim=2)
            return torch.mean(losses, dim=2, keepdim=True)
        else:
            return self.loss_fn(pred, target, normalize=self.normalize)

 