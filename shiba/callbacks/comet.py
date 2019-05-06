from .base import Callback


class Comet(Callback):
    def __init__(self, experiment):
        self.experiment = experiment

    def on_train_begin(self, state):
        self.experiment.set_name(state.get('experiment_name'))
        if state.get('hyperparams'):
            self.experiment.log_parameters(state.get('hyperparams'))

    def on_batch_end(self, state):
        step = state.get('step')
        self.experiment.log_metric('lr', state.get('learning_rate'), step)
        for metric, value in state['train_metrics'].items():
            self.experiment.log_metric(metric, value, step)

    def on_epoch_end(self, state):
        epoch = state.get('epoch')
        for metric, value in state['val_metrics'].items():
            self.experiment.log_metric(metric, value, epoch)

