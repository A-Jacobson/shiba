from torch.utils.tensorboard import SummaryWriter
from shiba.utils import get_lr, get_momentum

from .callbacks import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir=None, vis_function=None, hyperparams=None):
        self.vis_function = vis_function
        self.writer = None
        self.log_dir = log_dir
        self.hyperparams = hyperparams

    def on_train_begin(self, trainer):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.hyperparams:
            text = ''
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, trainer.global_step)

    def on_batch_end(self, trainer):
        self.writer.add_scalar('lr', get_lr(trainer.optimizer), trainer.global_step)
        if get_momentum(trainer.optimizer):
            self.writer.add_scalar('momentum', momentum, trainer.global_step)
        for metric, value in trainer.metrics.items():
            if 'train' in metric:
                self.writer.add_scalar(metric, value, trainer.global_step)

    def on_epoch_end(self, trainer):
        for metric, value in trainer.metrics.items():
            if 'val' in metric:
                self.writer.add_scalar(metric, value, trainer.global_step)

        if self.vis_function:
            vis = self.vis_function(trainer.val_out['inputs'],
                                    trainer.val_out['outputs'],
                                    trainer.val_out['targets'])
            for name, value in vis.items():
                self.writer.add_image(name, value, trainer.global_step)
