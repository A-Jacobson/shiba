from torch.utils.tensorboard import SummaryWriter

from .callbacks import Callback


class TensorBoard(Callback):
    def __init__(self, log_dir=None, snapshot_func=None, hyperparams=None):
        self.snapshot_func = snapshot_func
        self.writer = None
        self.log_dir = log_dir
        self.hyperparams = hyperparams

    def on_train_begin(self, state):
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.hyperparams:
            text = ''
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, state.logs.step)

    def on_batch_end(self, state):
        self.writer.add_scalar('lr', state.logs.lr, state.logs.step)
        if state.logs.momentum:
            self.writer.add_scalar('momentum', state.logs.momentum, state.logs.step)
        for metric, value in state.logs.metrics.items():
            if 'train' in metric:
                self.writer.add_scalar(metric, value, state.logs.step)

    def on_epoch_end(self, state):
        for metric, value in state.logs.metrics.items():
            if 'val' in metric:
                self.writer.add_scalar(metric, value, state.logs.step)

        if self.snapshot_func:
            val_out = state.logs.val_out
            inputs = val_out['inputs']
            outputs = val_out['outputs']
            targets = val_out['targets']
            snapshot = self.snapshot_func(inputs, outputs, targets)
            for name, value in snapshot.items():
                self.writer.add_image(name, value, state.logs.step)
