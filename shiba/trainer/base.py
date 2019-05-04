class Observable:
    def __init__(self):
        self.callbacks = []
        self.state = None

    def register_callbacks(self, callbacks):
        for callback in callbacks:
            self.register_callback(callback)

    def register_callback(self, callback):
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def remove_callback(self, callback):
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def remove_all(self):
        if self.callbacks:
            self.callbacks = []

    def epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(self.state)

    def epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self.state)

    def batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(self.state)

    def batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end(self.state)

    def eval_batch_begin(self):
        for callback in self.callbacks:
            callback.on_eval_batch_begin(self.state)

    def eval_batch_end(self):
        for callback in self.callbacks:
            callback.on_eval_batch_end(self.state)

    def eval_end(self):
        for callback in self.callbacks:
            callback.on_eval_end(self.state)

    def train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(self.state)

    def train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self.state)
        self.remove_all()
