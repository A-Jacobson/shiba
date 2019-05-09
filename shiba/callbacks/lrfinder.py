import copy

from shiba.utils import EndTraining, adjust_lr
from shiba.vis import plot_lr_find
from .callbacks import Callback


class LRFinder(Callback):

    def __init__(self, min_lr, max_lr):
        self.model_state = None
        self.optimizer_state = None
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_multiplier = None
        self.best_loss = 0
        self.losses = []
        self.lrs = []
        self.lr = min_lr

    def on_train_begin(self, state):
        core = state.core
        logs = state.logs
        self.model_state = copy.deepcopy(core.model.state_dict())
        self.optimizer_state = copy.deepcopy(core.optimizer.state_dict())
        self.lr_multiplier = (self.max_lr / self.min_lr) ** (1 / (logs.num_batches - 1))

    def on_batch_end(self, state):
        loss = state.logs.metrics['train_loss']
        if state.logs.step > 1 and loss > 4 * self.best_loss:
            plot_lr_find(self.lrs, self.losses)
            state.core.model.load_state_dict(self.model_state)
            state.core.optimizer.load_state_dict(self.optimizer_state)
            raise EndTraining('loss exploded, stop lr finder')

        else:
            self.best_loss = loss

        self.losses.append(loss)
        self.lrs.append(self.lr)
        self.lr *= self.lr_multiplier
        adjust_lr(state.core.optimizer, self.lr)

    def on_train_end(self, state):
        # restore states and plot results
        state.core.model.load_state_dict(self.model_state)
        state.core.optimizer.load_state_dict(self.optimizer_state)
        plot_lr_find(self.lrs, self.losses)
