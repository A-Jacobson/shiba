from tqdm.auto import tqdm

from .callbacks import Callback


class ProgressBar(Callback):
    def __init__(self):
        self.train_pbar = None
        self.epoch_pbar = None

    def on_train_begin(self, trainer):
        self.train_pbar = tqdm(total=trainer.epochs, unit='epochs')

    def on_epoch_begin(self, trainer):
        self.epoch_pbar = tqdm(total=trainer.num_batches, unit='b')

    def on_epoch_end(self, trainer):
        self.train_pbar.update()
        self.epoch_pbar.close()

    def on_batch_end(self, trainer):
        self.epoch_pbar.update()
        self.epoch_pbar.set_postfix(trainer.metrics)

    def on_eval_end(self, trainer):
        self.epoch_pbar.set_postfix(trainer.metrics)

    def on_train_end(self, trainer):
        self.train_pbar.close()
