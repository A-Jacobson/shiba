from tqdm.auto import tqdm

from .callbacks import Callback


class ProgressBar(Callback):
    def __init__(self):
        self.train_pbar = None
        self.epoch_pbar = None

    def on_train_begin(self, state):
        epochs = state.logs.epochs
        self.train_pbar = tqdm(range(epochs), total=epochs, unit='epochs')

    def on_epoch_begin(self, state):
        self.epoch_pbar = tqdm(total=state.logs.num_batches, unit='b')

    def on_epoch_end(self, state):
        self.train_pbar.update()
        self.epoch_pbar.close()

    def on_batch_end(self, state):
        self.epoch_pbar.update()
        self.epoch_pbar.set_postfix(state.logs.metrics)

    def on_eval_end(self, state):
        self.epoch_pbar.set_postfix(state.logs.metrics)

    def on_train_end(self, state):
        self.train_pbar.close()
