from .callbacks import Callback


# idea from kekeas
class Debug(Callback):
    """
    inspect training at different points in the loop
    possible entry points: ('on_epoch_begin', 'on_epoch_end',
                           'on_batch_begin', 'on_batch_end',
                           'on_eval_batch_begin', 'on_eval_batch_end',
                           'on_train_begin', 'on_train_end',
                           'on_eval_end')
    """

    def __init__(self, events):
        self.events = events
        possible_events = ('on_epoch_begin', 'on_epoch_end',
                           'on_batch_begin', 'on_batch_end',
                           'on_eval_batch_begin', 'on_eval_batch_end',
                           'on_train_begin', 'on_train_end',
                           'on_eval_end')
        for e in events:
            if e not in possible_events:
                raise ValueError(f'{e} not a valid event, must be one of {possible_events}')

    def on_epoch_begin(self, state):
        if 'on_epoch_begin' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_epoch_end(self, state):
        if 'on_epoch_end' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_batch_begin(self, state):
        if 'on_batch_begin' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_batch_end(self, state):
        if 'on_batch_end' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_eval_batch_begin(self, state):
        if 'on_eval_batch_begin' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_eval_batch_end(self, state):
        if 'on_eval_batch_end' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_train_begin(self, state):
        if 'on_train_begin' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_train_end(self, state):
        if 'on_train_end' in self.events:
            from ipdb import set_trace
            set_trace()

    def on_eval_end(self, state):
        if 'on_eval_end' in self.events:
            from ipdb import set_trace
            set_trace()
