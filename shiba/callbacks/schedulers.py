import copy

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from .callbacks import Callback
from shiba import schedulers
import torch


class PytorchScheduler(Callback):
    """Wraps a pytorch scheduler."""
    # TODO refactor this for OneCycle API
    def __init__(self, scheduler, update_interval='batch'):
        self.scheduler = scheduler
        if update_interval not in ('batch', 'epoch'):
            raise ValueError('update must be "batch" or "epoch"')
        self.update_interval = update_interval

    def on_batch_end(self, state):
        if self.update_interval == 'batch':
            self.scheduler.step()

    def on_epoch_end(self, state):
        if self.update_interval == 'epoch':
            self.scheduler.step()

    def simulate(self, steps=1000):
        scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        optimizer_state = copy.deepcopy(self.scheduler.optimizer.state_dict())

        lrs = []
        for i in range(steps):
            lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            lrs.append(lr)

        self.scheduler.load_state_dict(scheduler_state)
        self.scheduler.optimizer.load_state_dict(optimizer_state)

        plt.plot(lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel(f'{self.update_interval}')
        plt.ylabel('learning rate')
        plt.show()


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    """

    def __init__(self, monitor='val_loss',
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=0.0001,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-08):
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.scheduler = None

    def on_train_begin(self, state):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(state.core.optimizer,
                                                        mode=self.mode, factor=self.factor, patience=self.patience,
                                                        verbose=False, threshold=self.threshold,
                                                        threshold_mode=self.threshold_mode,
                                                        cooldown=self.cooldown, min_lr=self.min_lr, eps=self.eps)

    def on_epoch_end(self, state):
        value = state.logs.metrics.get(self.monitor)
        if not value:
            raise ValueError(
                f'could not find metric: {self.monitor} track it with a callback: `Metric(score_func, {self.monitor})`!')
        self.scheduler.step(value)


class OneCycle(Callback):
    # adapted from https://github.com/titu1994/keras-one-cycle/blob/master/clr.py
    def __init__(self,
                 max_lr,
                 end_percentage=0.1,
                 scale_percentage=None,
                 maximum_momentum=0.95,
                 minimum_momentum=0.85):
        """ This callback implements a cyclical learning rate policy (CLR).
        This is a special case of Cyclic Learning Rates, where we have only 1 cycle.
        After the completion of 1 cycle, the learning rate will decrease rapidly to
        100th its initial lowest value.
        # Arguments:
            max_lr: Float. Initial learning rate. This also sets the
                starting learning rate (which will be 10x smaller than
                this), and will increase to this value during the first cycle.
            end_percentage: Float. The percentage of all the epochs of training
                that will be dedicated to sharply decreasing the learning
                rate after the completion of 1 cycle. Must be between 0 and 1.
            scale_percentage: Float or None. If float, must be between 0 and 1.
                If None, it will compute the scale_percentage automatically
                based on the `end_percentage`.
            maximum_momentum: Optional. Sets the maximum momentum (initial)
                value, which gradually drops to its lowest value in half-cycle,
                then gradually increases again to stay constant at this max value.
                Can only be used with SGD Optimizer.
            minimum_momentum: Optional. Sets the minimum momentum at the end of
                the half-cycle. Can only be used with SGD Optimizer.
        # Reference
            - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, weight_decay, and weight decay](https://arxiv.org/abs/1803.09820)
            - [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
        """
        self.max_lr = max_lr
        self.end_percentage = end_percentage
        self.scale_percentage = float(scale_percentage) if scale_percentage is not None else float(end_percentage)
        self.max_momentum = maximum_momentum
        self.min_momentum = minimum_momentum
        self.scheduler = None

    def on_train_begin(self, state):
        self.scheduler = schedulers.OneCycle(optimizer=state.core.optimizer, max_lr=self.max_lr,
                                             end_percentage=self.end_percentage, scale_percentage=self.scale_percentage,
                                             maximum_momentum=self.max_momentum, minimum_momentum=self.min_momentum,
                                             num_iterations=state.logs.num_batches * state.logs.epochs)

        self.scheduler.setup()

    def on_batch_end(self, state):
        self.scheduler.step()

    def simulate(self, steps=1000):
        m = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(m.parameters(), lr=1)
        self.scheduler = schedulers.OneCycle(optimizer=optimizer, max_lr=self.max_lr,
                                             end_percentage=self.end_percentage, scale_percentage=self.scale_percentage,
                                             maximum_momentum=self.max_momentum, minimum_momentum=self.min_momentum,
                                             num_batches=steps)

        lrs = []
        moms = []
        self.scheduler.setup()
        for i in range(steps):
            lr = self.scheduler.optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if self.scheduler.use_momentum:
                mom = self.scheduler.optimizer.param_groups[0]['momentum']
                moms.append(mom)
            self.scheduler.step()
        if moms:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8), sharex=True)
            ax1.set_title('Learning Rate Schedule')
            ax1.plot(lrs)
            ax1.set_xlabel(f'batch')
            ax1.set_ylabel('learning rate')
            ax2.set_title('Momentum Schedule')
            ax2.plot(moms)
            ax2.set_xlabel('batch')
            ax2.set_ylabel('momentum')
            ax1.grid()
            ax2.grid()
            plt.tight_layout()
        else:
            plt.figure(figsize=(8, 4))
            plt.plot(lrs)
            plt.title('Learning Rate Schedule')
            plt.xlabel(f'batch')
            plt.ylabel('learning rate')
            plt.grid()
            plt.tight_layout()
        plt.show()

