from pathlib import Path

import torch

from shiba.callbacks import Callback


class Save(Callback):
    def __init__(self, save_dir, monitor='val_loss', mode='min', interval=None, max_saves=2):
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.interval = interval
        self.max_saves = max_saves
        self.last_save = 0
        if mode not in ('min', 'max'):
            raise ValueError(f'mode must be "min" or "max.')
        self.mode = mode
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = -float('inf')
        self.value = None

    def on_epoch_end(self, state):
        self.last_save += 1
        core = state.core
        logs = state.logs
        self.value = logs.metrics.get(self.monitor) or logs.train_metrics
        value = self.value if self.mode == 'min' else -self.value  # flip comparison if mode = max
        if self.last_save == self.interval and value > self.best_value:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = dict(optimizer_state=core.optimizer.state_dict(),
                              model_state=core.model.state_dict(),
                              logs=logs)
            torch.save(checkpoint, self.save_dir / f'epoch:{logs.epoch}_{self.monitor}:{self.value:.3f}.pth')
            self.last_save = 0
            self.best_value = value
        else:
            if not value:
                raise ValueError(f'counting find metric: {self.monitor} make sure to track it!')
