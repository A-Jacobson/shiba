import neptune
from shiba.utils import get_lr, get_momentum
from shiba.vis import plot_confusion_matrix

from shiba.callbacks import Callback
from shiba.callbacks.confusion import ConfusionMatrix


class NeptuneCallback(Callback):
    def __init__(self, project, exp_name, vis_function=None, hyperparams=None):
        self.project = project
        self.exp_name = exp_name
        self.hyperparams = hyperparams
        self.vis_function = vis_function
        neptune.init(self.project)
        neptune.create_experiment(self.exp_name, params=self.hyperparams)

    def on_batch_end(self, trainer):
        neptune.log_metric('learning_rate', get_lr(trainer.optimizer))
        momentum = get_momentum(trainer.optimizer)
        if momentum:
            neptune.log_metric('momentum', momentum)
        for metric, value in trainer.metrics.items():
            if 'train' in metric:
                neptune.log_metric(metric, value)

    def on_epoch_end(self, trainer):
        for metric, value in trainer.metrics.items():
            if 'val' in metric:
                neptune.log_metric(metric, value)

        if self.vis_function:
            vis = self.vis_function(trainer.out['inputs'],
                                    trainer.out['outputs'],
                                    trainer.out['targets'])
            for name, value in vis.items():
                neptune.log_image(name, value.cpu().numpy().transpose(1, 2, 0))

        cb = self.get_callback(trainer.callbacks, ConfusionMatrix)
        if cb:
            train_vis = plot_confusion_matrix(cb.train_matrix, cb.class_names, as_array=True)
            val_vis = plot_confusion_matrix(cb.val_matrix, cb.class_names, as_array=True)
            neptune.log_image('train_confusion_matrix', train_vis.cpu().numpy().transpose(1, 2, 0))
            neptune.log_image('val_confusion_matrix', val_vis.cpu().numpy().transpose(1, 2, 0))

    @staticmethod
    def get_callback(callbacks, callback):
        """
        return the first instance of a callback in the list of callbacks o(n)
        """
        cb = None
        try:
            cb = next(cb for cb in callbacks if isinstance(cb, callback))
        except StopIteration:
            pass
        return cb