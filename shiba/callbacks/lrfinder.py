import copy

from shiba.utils import EndTraining, adjust_lr
from shiba.vis import plot_lr_find

from .callbacks import Callback


class LRFinder(Callback):

    def __init__(self, model, optimizer, min_lr, max_lr, smoothing):
        self.model = model
        self.model_state = None
        self.optimizer_state = None
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smoothing = smoothing
        self.mult = None
        self.avg_loss = 0
        self.best_loss = 0
        self.losses = []
        self.lrs = []
        self.lr = min_lr

    def on_train_begin(self, state):
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.mult = (self.max_lr / self.min_lr) ** (1 / (state['num_batches'] - 1))

    def on_batch_end(self, state):
        self.avg_loss = self.smoothing * self.avg_loss + \
                        (1 - self.smoothing) * state['train_output']['loss']
        smoothed_loss = self.avg_loss / (1 - self.smoothing ** state['step'])

        if state['step'] > 1 and smoothed_loss > 4 * self.best_loss:
            raise EndTraining('loss exploded, stop lr finder')

        else:
            self.best_loss = smoothed_loss

        self.losses.append(smoothed_loss)
        self.lrs.append(self.lr)
        self.lr *= self.mult
        adjust_lr(self.optimizer, self.lr)

    def on_train_end(self, state):
        # restore states and plot results
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        plot_lr_find(self.lrs, self.losses)
