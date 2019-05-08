class Callback:
    """
    Base shiba callback, all possible events are defined here
    """

    def on_epoch_begin(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_batch_begin(self, state):
        pass

    def on_batch_end(self, state):
        pass

    def on_eval_batch_begin(self, state):
        pass

    def on_eval_batch_end(self, state):
        pass

    def on_train_begin(self, state):
        pass

    def on_train_end(self, state):
        pass

    def on_eval_end(self, state):
        pass


class Compose:
    """holds a group of callbacks calls their methods on events
    """

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def on_train_begin(self, state):
        for callback in self.callbacks:
            callback.on_train_begin(state)

    def on_epoch_begin(self, state):
        for callback in self.callbacks:
            callback.on_epoch_begin(state)

    def on_batch_begin(self, state):
        for callback in self.callbacks:
            callback.on_batch_begin(state)

    def on_batch_end(self, state):
        for callback in self.callbacks:
            callback.on_batch_end(state)

    def on_eval_batch_begin(self, state):
        for callback in self.callbacks:
            callback.on_batch_begin(state)

    def on_eval_batch_end(self, state):
        for callback in self.callbacks:
            callback.on_eval_batch_end(state)

    def on_eval_end(self, state):
        for callback in self.callbacks:
            callback.on_eval_end(state)

    def on_epoch_end(self, state):
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_train_end(self, state):
        for callback in self.callbacks:
            callback.on_train_end(state)

    def add_callback(self, callback):
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def add_callbacks(self, callbacks):
        callbacks = callbacks or []
        for callback in callbacks:
            self.add_callback(callback)
