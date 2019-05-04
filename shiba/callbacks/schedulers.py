from .base import Callback
from shiba.utils import adjust_lr


class LRScheduler(Callback):
    def __init__(self, scheduler, update='batch'):
        self.scheduler = scheduler
        if update not in ('batch', 'epoch'):
            raise ValueError('update must be "batch" or "epoch"')
        self.update = update

    def on_batch_end(self, state):
        if self.update == 'batch':
            self.scheduler.step()

    def on_epoch_end(self, state):
        if self.update == 'epoch':
            self.scheduler.step()


class LRFinder(Callback):
    def __init__(self, optimizer, min_lr=1e-7, max_lr=10):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.optimizer = optimizer
        self.losses = []
        self.learning_rates = []

    def on_batch_end(self, state):
        self.losses.append()
        mult = (self.max_lr / self.min_lr) ** (1/state.get('len_loader'))
        adjust_lr(self.optimizer, self.state['learning_rate'] * mult)
