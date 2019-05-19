import torch

from shiba.utils import ConfusionMeter
from .callbacks import Callback


class ConfusionMatrix(Callback):

    def __init__(self, num_classes, class_names=None, normalize=False):
        self.num_classes = num_classes
        self.class_names = class_names
        self.normalize = normalize
        self.train_meter = ConfusionMeter(num_classes=num_classes)
        self.val_meter = ConfusionMeter(num_classes=num_classes)

    @property
    def train_matrix(self):
        return self.train_meter.value(self.normalize)
        
    @property
    def val_matrix(self):
        return self.val_meter.value(self.normalize)

    @torch.no_grad()
    def on_batch_end(self, trainer):
        outputs = trainer.out['outputs']
        targets = trainer.out['targets']
        self.train_meter.update(outputs, targets)

    @torch.no_grad()
    def on_eval_batch_end(self, trainer):
        outputs = trainer.out['outputs']
        targets = trainer.out['targets']
        self.val_meter.update(outputs, targets)

    def on_epoch_begin(self, trainer):
        self.train_meter.reset()
        self.val_meter.reset()
