from torch.utils.tensorboard import SummaryWriter
from shiba.utils import get_lr, get_momentum

from .callbacks import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir=None, vis_function=None, hyperparams=None):
        self.vis_function = vis_function
        self.writer = None
        self.log_dir = log_dir
        self.hyperparams = hyperparams

    def on_train_begin(self, state):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.hyperparams:
            text = ''
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, state.logs.global_step)

    def on_batch_end(self, state):
        lr = get_lr(state.core.optimizer)
        momentum = get_momentum(state.core.optimizer)
        self.writer.add_scalar('lr', lr, state.logs.global_step)
        if state.logs.momentum:
            self.writer.add_scalar('momentum', momentum, state.logs.global_step)
        for metric, value in state.logs.metrics.items():
            if 'train' in metric:
                self.writer.add_scalar(metric, value, state.logs.global_step)

    def on_epoch_end(self, state):
        for metric, value in state.logs.metrics.items():
            if 'val' in metric:
                self.writer.add_scalar(metric, value, state.logs.global_step)

        if self.vis_function:
            val_out = state.core.val_out
            inputs = val_out['inputs']
            outputs = val_out['outputs']
            targets = val_out['targets']
            vis = self.vis_function(inputs, outputs, targets)
            for name, value in vis.items():
                self.writer.add_image(name, value, state.logs.global_step)
