from torch.utils.tensorboard import SummaryWriter

from .base import Callback


class TensorBoard(Callback):
    def __init__(self, experiment_name='', snapshot_func=None, hyperparams=None):
        self.snapshot_func = snapshot_func
        self.hyperparams = hyperparams
        self.writer = SummaryWriter(comment=f'_{experiment_name}')

    def on_train_begin(self, state):
        if self.hyperparams:
            text = ''
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, state.get('step'))

    def on_batch_end(self, state):
        step = state.get('step')
        self.writer.add_scalar('lr', state.get('learning_rate'), step)
        for metric, value in state['metrics'].items():
            if 'train' in metric:
                self.writer.add_scalar(metric, value, step)

    def on_epoch_end(self, state):
        step = state.get('step')
        for metric, value in state['metrics'].items():
            if 'val' in metric:
                self.writer.add_scalar(metric, value, step)

        if self.snapshot_func:
            step = state['step']
            val_output = state['val_output']
            inputs = val_output['inputs']
            outputs = val_output['outputs']
            targets = val_output['targets']
            snapshot = self.snapshot_func(inputs, outputs, targets)
            for name, value in snapshot.items():
                self.writer.add_image(name, value, step)
