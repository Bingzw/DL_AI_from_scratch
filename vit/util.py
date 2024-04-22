import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set seed for reproducibility
    :param seed: seed value
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Convert image to patches
    :param x: torch.Tensor representing the image of shape [B, C, H, W]
    :param path_size: number of pixels per dimension of the patches
    :param flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid
    :return: torch.Tensor representing the patches of shape [B, H'*W', C, p_H, p_W] or [B, H'*W', C*p_H*p_W]
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

