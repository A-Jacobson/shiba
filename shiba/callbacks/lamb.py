from .callbacks import Callback


class LambdaCallback(Callback):
    def __init__(self, on_train_begin=None, on_epoch_begin=None, on_batch_begin=None,
                 on_batch_end=None, on_eval_batch_begin=None, on_eval_batch_end=None,
                 on_eval_end=None, on_epoch_end=None, on_train_end=None):
        self.on_train_begin = on_train_begin
        self._on_epoch_begin = on_epoch_begin
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._on_eval_batch_begin = on_eval_batch_begin
        self._on_eval_batch_end = on_eval_batch_end
        self._on_eval_end = on_eval_end
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_epoch_begin(self, state):
        if self._on_epoch_begin:
            self._on_epoch_begin(state)

    def on_epoch_end(self, state):
        if self._on_epoch_end:
            self._on_epoch_end(state)

    def on_batch_begin(self, state):
        if self._on_batch_begin:
            self._on_batch_begin(state)

    def on_batch_end(self, state):
        if self._on_batch_end:
            self._on_batch_end(state)

    def on_eval_batch_begin(self, state):
        if self._on_eval_batch_begin:
            self._on_eval_batch_begin(state)

    def on_eval_batch_end(self, state):
        if self._on_eval_batch_end:
            self._on_eval_batch_end(state)

    def on_train_begin(self, state):
        if self._on_train_begin:
            self._on_train_begin(state)

    def on_train_end(self, state):
        if self._on_train_end:
            self._on_train_end(state)

    def on_eval_end(self, state):
        if self._on_eval_end:
            self._on_eval_end(state)
