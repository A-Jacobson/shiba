
class Callback:
    """
    Base shiba callback, all possible events are defined here
    """

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_batch_begin(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_eval_batch_begin(self, trainer):
        pass

    def on_eval_batch_end(self, trainer):
        pass

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_eval_end(self, trainer):
        pass


class Compose:
    """holds a group of callbacks calls their methods on events
    """

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_epoch_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)

    def on_batch_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer)

    def on_batch_end(self, trainer):
        for callback in self.callbacks:
            callback.on_batch_end(trainer)

    def on_eval_batch_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer)

    def on_eval_batch_end(self, trainer):
        for callback in self.callbacks:
            callback.on_eval_batch_end(trainer)

    def on_eval_end(self, trainer):
        for callback in self.callbacks:
            callback.on_eval_end(trainer)

    def on_epoch_end(self, trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

