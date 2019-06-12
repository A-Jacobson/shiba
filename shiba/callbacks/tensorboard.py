from tensorboardX import SummaryWriter
from shiba.utils import get_lr, get_momentum
from shiba.vis import plot_confusion_matrix

from .callbacks import Callback
from .confusion import ConfusionMatrix


class TensorBoard(Callback):
    def __init__(self, log_dir=None, vis_function=None, hyperparams=None):
        self.vis_function = vis_function
        self.writer = None
        self.log_dir = log_dir
        self.hyperparams = hyperparams

    def on_train_begin(self, trainer):
        self.writer = SummaryWriter(logdir=self.log_dir)
        if self.hyperparams:
            for name, value in self.hyperparams.items():
                self.writer.add_text('hyperparams', f'{name}: {str(value)}', trainer.global_step)

    def on_batch_end(self, trainer):
        self.writer.add_scalar('learning_rate', get_lr(trainer.optimizer), trainer.global_step)
        momentum = get_momentum(trainer.optimizer)
        if momentum:
            self.writer.add_scalar('momentum', momentum, trainer.global_step)
        for metric, value in trainer.metrics.items():
            if 'train' in metric:
                self.writer.add_scalar(metric, value, trainer.global_step)

    def on_epoch_end(self, trainer):
        for metric, value in trainer.metrics.items():
            if 'val' in metric:
                self.writer.add_scalar(metric, value, trainer.global_step)

        if self.vis_function:
            vis = self.vis_function(trainer.out['inputs'],
                                    trainer.out['outputs'],
                                    trainer.out['targets'])
            for name, value in vis.items():
                self.writer.add_image(name, value, trainer.global_step)

        cb = self.get_callback(trainer.callbacks, ConfusionMatrix)
        if cb:
            train_vis = plot_confusion_matrix(cb.train_matrix, cb.class_names, as_array=True)
            val_vis = plot_confusion_matrix(cb.val_matrix, cb.class_names, as_array=True)
            self.writer.add_image('train_confusion_matrix', train_vis, trainer.global_step)
            self.writer.add_image('val_confusion_matrix', val_vis, trainer.global_step)

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



