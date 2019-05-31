from pathlib import Path

from .callbacks import Callback


class Save(Callback):
    def __init__(self, save_dir, monitor='val_loss', mode='min', interval=1, max_saves=2):
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
        self.past_checkpoints = []

    def on_epoch_end(self, trainer):
        self.last_save += 1
        self.value = trainer.metrics.get(self.monitor)
        if not self.value:
            raise ValueError(
                f'could not find metric: {self.monitor} track it with a callback: `Metric(score_func, {self.monitor})`!')
        value = self.value if self.mode == 'min' else -self.value  # flip comparison if mode = max
        if (self.last_save == self.interval) and (value < self.best_value):
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / f'epoch:{trainer.epoch}-{self.monitor}:{self.value:.3f}.pth'
            trainer.save(save_path)
            self.past_checkpoints.append([value, save_path])
            # remove worst checkpoint before saving new checkpoint, also compare new checkpoint
            if len(self.past_checkpoints) > self.max_saves:
                if self.mode == 'min':
                    worst = max(self.past_checkpoints)
                else:
                    worst = min(self.past_checkpoints)
                self.past_checkpoints.remove(worst)
                value, path = worst
                trainer.save(save_path)
                Path(path).unlink()
            self.last_save = 0
            self.best_value = value
