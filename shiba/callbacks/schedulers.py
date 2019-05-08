from .callbacks import Callback


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
