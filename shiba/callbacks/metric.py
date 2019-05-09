import torch

from shiba.utils import AverageMeter
from .callbacks import Callback


class Metric(Callback):
    """Applies a function to targets and output at the end of each training and validation batch. Records the average.
    """

    def __init__(self, metric, name, output_transform=None):
        self.metric = metric
        self.name = name
        self.output_transform = output_transform
        self.train_history = []
        self.val_history = []
        self.train_score_meter = AverageMeter()
        self.val_score_meter = AverageMeter()

    @property
    def avg_train_score(self):
        return self.train_score_meter.avg

    @property
    def avg_val_score(self):
        return self.val_score_meter.avg

    @torch.no_grad()
    def on_batch_end(self, state):
        output = state.logs.train_out['outputs']
        targets = state.logs.train_out['targets']
        if self.output_transform:
            output = self.output_transform(output)
        score = self.metric(output, targets)
        self.train_score_meter.update(score.item())
        state.logs.metrics[f'train_{self.name}'] = self.avg_train_score  # update state

    @torch.no_grad()
    def on_eval_batch_end(self, state):
        output = state.logs.val_out['outputs']
        targets = state.logs.val_out['targets']

        if self.output_transform:
            output = self.output_transform(output)
        score = self.metric(output, targets)
        self.val_score_meter.update(score.item())
        state.logs.metrics[f'val_{self.name}'] = self.avg_val_score  # update state

    def on_epoch_end(self, state):
        self.train_history.append(self.avg_train_score)
        self.val_history.append(self.avg_val_score)
        self.train_score_meter.reset()
        self.val_score_meter.reset()

    def __repr__(self):
        return f'<Metric: ({self.name})>'
