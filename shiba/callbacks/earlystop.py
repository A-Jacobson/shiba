from shiba.callbacks import Callback
from shiba.utils import EndTraining


class EarlyStop(Callback):
    def __init__(self, monitor='val_loss', mode='min', patience=3):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.miss = 0
        if mode not in ('min', 'max'):
            raise ValueError(f'mode must be "min" or "max.')
        self.mode = mode
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = -float('inf')
        self.value = None

    def on_epoch_end(self, trainer):
        self.value = trainer.logs.metrics.get(self.monitor)
        if not self.value:
            raise ValueError(
                f'could not find metric: {self.monitor} track it with a callback: `Metric(score_func, {self.monitor})`!')
        value = self.value if self.mode == 'min' else -self.value  # flip comparison if mode = max
        if value > self.best_value:
            self.miss = 0
            self.best_value = value
        elif value > self.best_value and self.miss >= self.patience:
            raise EndTraining('Early Stopping...')
        else:
            self.miss += 1

