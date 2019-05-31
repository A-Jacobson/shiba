import math
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from PIL import Image
from torchvision.utils import make_grid
import itertools

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


def show_images(images, num_columns=4, titles=None, scale=6, as_array=False, title_colors=None):
    """
    Arguments:
        images (list): list of images.
        num_columns (int): number of columns.
        titles (list, optional): list of image titles
    """
    title_colors = title_colors or ['b'] * len(images)
    if isinstance(images[0], Image.Image):
        images = [np.array(image.copy()) for image in images]
    if isinstance(images[0], torch.Tensor):
        images = [image.permute(1, 2, 0) if image.dim() == 3 else image.float() for image in images]

    num_rows = math.ceil(len(images) / num_columns)
    figure, axes = plt.subplots(nrows=num_rows, ncols=num_columns,
                             figsize=(num_columns * scale, num_rows * scale))

    blank_image = np.zeros_like(images[0])
    for i, axis in enumerate(axes.ravel()):
        if i >= len(images):
            axis.imshow(blank_image)
            if titles:
                axis.set_title('', color='bl')
        else:
            axis.imshow(images[i])
            if titles:
                axis.set_title(titles[i], color=title_colors[i])
        axis.axis('off')

    figure.subplots_adjust(hspace=0.08, wspace=0)
    if titles:
        figure.subplots_adjust(hspace=0.08, wspace=0)
    plt.tight_layout()
    if as_array:
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches="tight")
        plt.close(figure)
        buff.seek(0)
        return torch.from_numpy(np.array(Image.open(buff))).permute(2, 0, 1)


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


def vis_segment(inputs, outputs, targets, nrow=4):
    outputs_grid = make_grid(outputs.sigmoid() > 0.5)
    inputs_grid = make_grid(inputs[:3, ...], nrow=nrow)
    targets_grid = make_grid(targets, nrow=nrow)
    predictions = apply_masks(inputs_grid, outputs_grid)
    targets = apply_masks(inputs_grid, targets_grid)
    return dict(preddictions=predictions, targets=targets)


def vis_classify(inputs, outputs, targets, class_names=None, num_columns=4, scale=6):
    predictions = outputs.argmax(dim=1)
    titles = []
    title_colors = []
    for pred, target in zip(predictions, targets):
        pred, target = pred.item(), target.item()
        if class_names:
            pred, target = class_names[pred], class_names[target]
        titles.append(f'pred: {pred}\ntarget: {target}')
        title_colors.append('b' if pred == target else 'r')
    grid = show_images(inputs[:, :3, ...].cpu(), num_columns=num_columns,
                       titles=titles, title_colors=title_colors,
                       as_array=True, scale=scale)
    return dict(vis_classify=grid)


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


def plot_confusion_matrix(cm, class_names=None, as_array=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    class_names = class_names or np.arange(len(cm))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    threshold = cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    if as_array:
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches="tight")
        plt.close(figure)
        buff.seek(0)
        return torch.from_numpy(np.array(Image.open(buff))).permute(2, 0, 1)


def plot_text(outputs, targets, i2w=None, limit=5, as_array=False):
    import seaborn as sns
    targets = targets.t().cpu()[:limit]
    preds = outputs.argmax(dim=-1).t().cpu()[:limit]
    diff = (targets != preds)
    figure, axes = plt.subplots(nrows=limit,
                                figsize=(targets.shape[1], limit), sharex=True)

    def to_words(indices, i2w=None):
        return np.array([i2w[i] for i in indices])

    for i in range(limit):

        diff = np.stack([diff[i], diff[i]])  # highlight mis-predicted words
        if i2w:
            text = np.stack([to_words(targets[i], i2w), # if dict, show words
                             to_words(preds[i], i2w)])
        else:
            text = np.stack([targets[i], preds[i]])  # else show word indices
        sns.heatmap(diff, cmap=plt.cm.Purples, annot=text, fmt='s', cbar=False,
                    yticklabels=[f'target\n{i}  ', f'pred\n{i}  '],
                    xticklabels=False, ax=axes[i])
        plt.sca(axes[i])
        plt.yticks(rotation=0)
    plt.tight_layout()

    if as_array:
        buff = io.BytesIO()
        plt.savefig(buff, format='png', bbox_inches="tight")
        plt.close(figure)
        buff.seek(0)
        return torch.from_numpy(np.array(Image.open(buff))).permute(2, 0, 1)


def vis_text(inputs, outputs, targets, i2w=None, limit=5):
    return dict(vis_text=plot_text(outputs, targets, i2w, limit, as_array=True))
