from shiba.utils import get_lr, get_momentum

from .callbacks import Callback


class PolyaxonLogger(Callback):
    def __init__(self, experiment=None, hyperparams=None):
        self.experiment = experiment
        self.hyperparams = hyperparams

    def on_train_begin(self, trainer):
        if self.hyperparams:
            self.experiment.log_params(**self.hyperparams)

    def on_batch_end(self, trainer):
        self.experiment.log_metrics(**{'lr': get_lr(trainer.optimizer), 'step': trainer.global_step})
        momentum = get_momentum(trainer.optimizer)
        if momentum:
            self.experiment.log_metrics(**{'momentum': momentum, 'step': trainer.global_step})
        for metric, value in trainer.metrics.items():
            if 'train' in metric:
                self.experiment.log_metrics(**{metric: value, 'step': trainer.global_step})

    def on_epoch_end(self, trainer):
        for metric, value in trainer.metrics.items():
            if 'val' in metric:
                self.experiment.log_metrics(**{metric: value, 'step': trainer.global_step})
