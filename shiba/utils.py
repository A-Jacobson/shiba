from shutil import copyfile

import torch
from torchvision import transforms


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



def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


