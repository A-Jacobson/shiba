import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import ticker
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid

COLORS = [(94, 79, 162),  # purple
          (50, 136, 189),  # purple
          (102, 194, 165),  # orange
          (171, 221, 164),  # green
          (230, 245, 152),  # red
          (255, 255, 191),  # yellow
          (254, 224, 139),  # dark blue
          (253, 174, 97),
          (244, 109, 67),
          (213, 62, 79),
          (158, 1, 66)]


def plot_color_legend(labels, colors=COLORS):
    pal = [tuple(v / 255. for v in c) for c in colors]
    n = len(labels)
    f, ax = plt.subplots(1, 1, figsize=(n * 1, 1))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=plt.cm.colors.ListedColormap(list(pal[:n])),
              interpolation="nearest", aspect="auto")
    ax.set_yticks([-.5, .5])
    l = [''] + labels  # hack because maplotlib2.0 cuts off one label
    ax.set_xticklabels(l, rotation=45)
    ax.set_yticklabels([])
    f.subplots_adjust(bottom=0.15)
    return ax


def show_images(images, num_columns=4, titles=None, scale=6):
    """
    Arguments:
        images (list): list of images.
        num_columns (int): number of columns.
        titles (list, optional): list of image titles
    """
    if isinstance(images[0], Image.Image):
        images = [np.array(image.copy()) for image in images]
    if isinstance(images[0], torch.Tensor):
        images = [image.permute(1, 2, 0) if image.dim() == 3 else image.float() for image in images]

    num_rows = math.ceil(len(images) / num_columns)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns,
                             figsize=(num_columns * scale, num_rows * scale))

    blank_image = np.zeros_like(images[0])
    for i, axis in enumerate(axes.ravel()):
        if i >= len(images):
            axis.imshow(blank_image)
            if titles:
                axis.set(title='')
        else:
            axis.imshow(images[i])
            if titles:
                axis.set(title=titles[i])
        axis.axis('off')

    fig.subplots_adjust(hspace=0.08, wspace=0)
    if titles:
        fig.subplots_adjust(hspace=0.08, wspace=0)


def _apply_mask(image, mask, color_rgb=(66, 244, 223), alpha=0.7):
    """
    Args:
        image: (uint8) numpy array of shape (h, w, c)
        mask: (uint8) numpy array of shape (h, w)
        color_rgb:
        alpha:

    Returns:

    """
    color = np.array(color_rgb, dtype='uint8')
    masked_image = np.copy(image)
    masked_image[mask > 0] = image[mask > 0] * (1 - alpha) + alpha * color
    return masked_image


def apply_masks(image, masks, colors=COLORS, alpha=0.7):
    """
    Args:
        image: numpy array or torch tensor (h, w, c) or (c, h, w)
        mask: numpy array or torch tensor (h, w, c) or (c, h, w)
        colors: (list of tuples), color palette (0-1)
        alpha: transparency of the masks
    Returns:
        new image with masks applied
    """

    def format_shape(array):
        if len(array.shape) != 3:
            raise ValueError('images and masks must be 3-dimensional')
        channel_index = np.argmin(array.shape)
        if channel_index != 2:
            array = array.transpose(1, 2, channel_index)
        return array

    def format_image_dtype(image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.dtype != 'uint8':
            image = (image * 255.).astype('uint8')
        return image

    def format_mask_dtype(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.dtype != 'uint8':
            mask = mask.astype('uint8')
        return mask

    image = format_image_dtype(image)
    masks = format_mask_dtype(masks)
    image = format_shape(image)
    masks = format_shape(masks)

    for i in range(masks.shape[-1]):
        image = _apply_mask(image, masks[..., i], color_rgb=colors[i % len(colors)], alpha=alpha)
    return image


def annotate_tensor(tensor, text, color=(255, 255, 255), size=None, position=None):
    scale = int(256 / tensor.shape[1])
    if not size:
        size = int(12 / scale) + 6
    if not position:
        position = (int(10 / scale), int(10 / scale))
    image = to_pil_image(tensor)
    font = ImageFont.truetype('shiba/assets/Roboto-Bold.ttf', size=size)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, color, font)
    return to_tensor(image)


def segmentation_snapshot(inputs, outputs, targets, nrow=4):
    outputs_grid = make_grid(outputs.sigmoid() > 0.5)
    inputs_grid = make_grid(inputs[:3, ...], nrow=nrow)
    targets_grid = make_grid(targets, nrow=nrow)
    predictions = apply_masks(inputs_grid, outputs_grid)
    targets = apply_masks(inputs_grid, targets_grid)
    return dict(preddictions=predictions, targets=targets)


def classification_snapshot(inputs, outputs, targets, nrow=4):
    predictions = outputs.argmax(dim=1)
    annotated = []
    for image, pred, target in zip(inputs, predictions, targets):
        text = f'p: {pred}, t:{target}'
        color = (0, 255, 0)  # green
        if pred != target:
            color = (255, 0, 0)  # red
        annotated.append(annotate_tensor(image, text, color))
    grid = make_grid(annotated, nrow=nrow)
    return dict(snapshot=grid)


def plot_lr_find(lrs, losses):
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x:1.0e}')
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(lrs[10:-5], losses[10:-5])
    plt.title('Learning Rate Finder')
    plt.ylabel('Loss')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.grid()
    plt.show()
