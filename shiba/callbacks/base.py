
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


