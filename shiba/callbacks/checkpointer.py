import torch
from shiba.callbacks import Callback


class CheckPointer(Callback):
    # TODO
    def __init__(self, experiment_name, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, state):
        checkpoint = {'model_state': self.model.state_dict(),
                      'optimizer_state': self.optimizer.state_dict(),
                      'hyperparams': state.get('hyperparams')}
        torch.save(checkpoint, f'{self.experiment_name}.pt')
