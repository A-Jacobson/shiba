import numpy as np
import torch
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from torchvision.transforms import functional as F


def mask_to_tensor(mask):
    """
    transforms (H, W) or (H, W, Class) numpy array into (Class, H, W) tensor
    """
    mask = torch.from_numpy(mask).float()
    if mask.ndimension() == 2:
        mask = mask.unsqueeze(-1)
    return mask.permute(2, 0, 1)


def numpy_to_tensor(image):
    """
    transform (H, W, C) to (C, H, W) and scale between 0 and 1.
    """
    image = image / (255. if image.dtype == np.uint8 else 1)
    image = torch.from_numpy(image.astype(np.float32))
    return image.permute(2, 0, 1)


class ToTensor(DualTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True)

    def apply(self, image, **params):
        return numpy_to_tensor(image)

    def apply_to_mask(self, mask, **params):
        return mask_to_tensor(mask)


class Normalize(ImageOnlyTransform):
    def __init__(self, mean, std):
        super(Normalize, self).__init__(always_apply=True)
        self.mean = mean
        self.std = std

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std)
