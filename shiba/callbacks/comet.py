from .base import Callback


class Comet(Callback):
    def __init__(self, experiment, experiment_name, hyperparams=None):
        self.experiment = experiment
        self.experiment_name = experiment_name
        self.hyperparams = hyperparams

    def on_train_begin(self, state):
        self.experiment.set_name(self.experiment_name)
        if self.hyperparams:
            self.experiment.log_parameters(self.hyperparams)

    def on_batch_end(self, state):
        step = state.get('step')
        self.experiment.log_metric('lr', state.get('learning_rate'), step)
        for metric, value in state['metrics'].items():
            if 'train' in metric:
                self.experiment.log_metric(metric, value, step)

    def on_epoch_end(self, state):
        epoch = state.get('epoch')
        for metric, value in state['metrics'].items():
            if 'val' in metric:
                self.experiment.log_metric(metric, value, epoch)

