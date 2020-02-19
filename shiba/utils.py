import numpy as np
import torch
from torch import nn
from torchvision import transforms
from collections import OrderedDict


class EndTraining(Exception):
    """
    Raised to break out of shiba training loop
    """
    pass


def imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def gram_matrix(img):
    (b, ch, h, w) = img.size()
    features = img.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features @ features_t / (ch * h * w)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


class ExponentialAverage:
    def __init__(self, smoothing=0.98):
        self.avg = 0
        self.sum = 0
        self.smoothing = smoothing
        self.count = 0
        self.value = 0

    def update(self, value):
        self.value = value
        self.count += 1
        self.sum = self.smoothing * self.sum + \
                   (1 - self.smoothing) * value
        self.avg = self.sum / (1 - self.smoothing ** self.count)

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_momentum(optimizer, momentum):
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def count_parameters(module):
    total_params = 0
    for p in module.parameters():
        if str(p.layout) == 'torch.sparse_coo':
            total_params += p._nnz()
        else:
            if p.requires_grad:
                total_params += p.numel()
    return total_params


class LayerCache:
    """
    Saves the input and output of an `nn.Moldule` for further inspection.

    >>> unet = ResUnet34(in_channels=3, out_channels=3, num_filters=32)
    >>> layer = LayerCache(unet.decode_5)
    >>> out = unet(x) # run model, layercache saves activations resulting from this run
    >>> print(layer.output.shape) # torch.Size([1, 256, 16, 16])
    >>> layer.handle.remove()
    """

    def __init__(self, module, name=None):
        self.module = module
        self.name = name
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.module = module
        self.input = input
        self.output = output


def model_summary(model, *inputs, markdown=True, return_layers=False):
    from tabulate import tabulate
    from IPython.display import display, Markdown

    layers = []
    for name, module in model.named_children():
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for n, m in module.named_children():
                layers.append(LayerCache(m, f'{name}-{n}'))
        else:
            layers.append(LayerCache(module, name))

    out = model(*inputs)
    summary = []
    total_params = 0
    for layer in layers:
        params = count_parameters(layer.module)
        total_params += params
        output = layer.output
        if isinstance(output, tuple):
            shape = [tuple(o[-1].shape) for o in output]
        else:
            shape = tuple(output.shape)
        summary.append([layer.name, shape, f'{params:,}'])
        layer.handle.remove()
    summary.append(['TOTAL:', '  -----------------  ', f'{total_params:,}'])

    table = tabulate(summary,
                     headers=('Name', 'Output Size', 'Parameters'),
                     disable_numparse=True,
                     tablefmt='pipe')
    if markdown:
        display(Markdown(table))
    else:
        print(table)

    if return_layers:
        return layers


def model_to_devices(model, device, device_ids: tuple = -1):
    model = model.to(device)
    num_devices = torch.cuda.device_count()
    if device_ids == -1 and num_devices > 1:
        model = torch.nn.DataParallel(model)
    if device_ids != -1 and num_devices > 1:
        model = torch.nn.DataParallel(model, device_ids)
    return model


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_momentum(optimizer):
    return optimizer.param_groups[0].get('momentum')


class ConfusionMeter:

    def __init__(self, num_classes):
        super(ConfusionMeter, self).__init__()
        self.matrix = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix.fill(0)

    def update(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        predicted = predicted.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.num_classes ** 2)
        assert bincount_2d.size == self.num_classes ** 2
        matrix = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.matrix += matrix

    def value(self, normalized=False):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if normalized:
            matrix = self.matrix.astype(np.float32)
            return np.around(matrix / matrix.sum(axis=1)[:, None], decimals=4)
        else:
            return self.matrix


def remove_hooks(model):
    if model._backward_hooks != OrderedDict():
        print('removing backward hooks: ', model._backward_hooks)
        model._backward_hooks = OrderedDict()
    if model._forward_hooks != OrderedDict():
        print('removing forward hooks: ', model._forward_hooks)
        model._forward_hooks = OrderedDict()
    if model._forward_pre_hooks != OrderedDict():
        print('removing forward pre-hooks: ', model._forward_pre_hooks)
        model._forward_pre_hooks = OrderedDict()
    for child in model.children():
        remove_hooks(child)
