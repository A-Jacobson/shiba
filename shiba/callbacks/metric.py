import torch

from shiba.utils import AverageMeter, ExponentialAverage
from .callbacks import Callback


class Metric(Callback):
    """Applies a function to train_step and eval_step outputs at the end of each training and validation batch.
    Retains per-epoch history of scores.
    """

    def __init__(self, name, score_func=None, transform=lambda x: (x['outputs'], x['targets']), train_smoothing=0.98):
        """
        Args:
            name: metrics will be saved to `trainer.logs.metrics['train_{name}']` and `trainer.logs.metrics['val_{name}']`
            metric: function to apply to processed output such as `accuracy`
            transform: function that prepares train_step output for metric function
            train_smoothing: logs training metric with ExponentialAverage
        """
        self.score_func = score_func
        self.name = name
        self.transform = transform
        self.train_history = []
        self.val_history = []
        if train_smoothing:
            self.train_meter = ExponentialAverage(smoothing=train_smoothing)
        else:
            self.train_meter = AverageMeter()
        self.val_meter = AverageMeter()

    @torch.no_grad()
    def on_batch_end(self, trainer):
        score = self.transform(trainer.train_out)
        if self.score_func:
            if isinstance(score, tuple):
                score = self.score_func(*score)
            else:
                score = self.score_func(score)
        self.train_meter.update(score)
        trainer.metrics[f'train_{self.name}'] = self.train_meter.avg  # update trainer

    @torch.no_grad()
    def on_eval_batch_end(self, trainer):
        score = self.transform(trainer.val_out)
        if self.score_func:
            if isinstance(score, tuple):
                score = self.score_func(*score)
            else:
                score = self.score_func(score)
        self.val_meter.update(score)
        trainer.metrics[f'val_{self.name}'] = self.val_meter.avg  # update trainer

    def on_epoch_end(self, trainer):
        self.train_history.append(self.train_meter.avg)
        self.val_history.append(self.val_meter.avg)
        self.train_meter.reset()
        self.val_meter.reset()

    def __repr__(self):
        return f'<Metric: ({self.name})>'
