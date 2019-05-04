from torch.utils.tensorboard import SummaryWriter
from .base import Callback


class TensorBoard(Callback):
    def __init__(self, experiment_name='', snapshot_func=None, hyperparams=None):
        self.snapshot_func = snapshot_func
        self.hyperparams = hyperparams
        self.writer = SummaryWriter(comment=experiment_name)

    def on_train_begin(self, state):
        if self.hyperparams:
            text = ""
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, state.get('epoch'))

    def on_epoch_end(self, state):
        epoch = state.get('epoch')
        for metric, value in state['metrics'].items():
            self.writer.add_scalar(metric, value, epoch)

        if self.snapshot_func:
            epoch = state['epoch']
            val_output = state['val_output']
            inputs = val_output['inputs']
            outputs = val_output['outputs']
            targets = val_output['targets']
            snapshot = self.snapshot_func(inputs, outputs, targets)
            for name, value in snapshot.items():
                self.writer.add_image(name, value, epoch)

