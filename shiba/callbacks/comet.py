from .callbacks import Callback


class Comet(Callback):
    def __init__(self, experiment, name=None, hyperparams=None):
        self.experiment = experiment
        self.name = name
        self.hyperparams = hyperparams

    def on_train_begin(self, trainer):
        self.experiment.set_name(self.name)
        self.experiment.log_parameters(self.hyperparams)

    def on_batch_end(self, trainer):
        self.experiment.log_metric('lr', trainer.logs.lr, step)
        for metric, value in trainer.logs.metrics.items():
            if 'train' in metric:
                self.experiment.log_metric(metric, value, step)

    def on_epoch_end(self, trainer):
        step = trainer.logs.global_step
        for metric, value in trainer.logs.metrics.items():
            if 'val' in metric:
                self.experiment.log_metric(metric, value, step)
