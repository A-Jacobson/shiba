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
        self._step_save = None

    def plot(self):
        plot_lr_find(self.lrs, self.losses)

    def on_train_begin(self, trainer):
        self._step_save = trainer.global_step
        self.model_state = copy.deepcopy(trainer.model.state_dict())
        self.optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
        self.lr_multiplier = (self.max_lr / self.min_lr) ** (1 / (trainer.num_batches - 1))

    def on_batch_end(self, trainer):
        loss = trainer.metrics['train_loss']  # smoothed loss from metrics
        if trainer.step > 1 and loss > 4 * self.best_loss:
            plot_lr_find(self.lrs, self.losses)
            trainer.model.load_state_dict(self.model_state)
            trainer.optimizer.load_state_dict(self.optimizer_state)
            raise EndTraining('loss exploded, stop lr finder')

        else:
            self.best_loss = loss

        self.losses.append(loss)
        self.lrs.append(self.lr)
        self.lr *= self.lr_multiplier
        adjust_lr(trainer.optimizer, self.lr)

    def on_train_end(self, trainer):
        # restore states and plot results
        trainer.model.load_state_dict(self.model_state)
        trainer.optimizer.load_state_dict(self.optimizer_state)
        trainer.global_step = self._step_save
        plot_lr_find(self.lrs, self.losses)
