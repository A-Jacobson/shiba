import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

from shiba.callbacks import Compose
from shiba.callbacks import Metric, ProgressBar, LRFinder
from shiba.utils import adjust_lr, DotDict, EndTraining


class Trainer:
    """Shiba Trainer"""

    def __init__(self, model, criterion,
                 optimizer=None, train_step=None, eval_step=None):
        """Example of docstring on the __init__ method.
        Args:
            model: pytorch model.
            criterion: loss function.
            optimizer: pytorch optimizer. defaults to Adam
            train_dataset: pytorch training dataset.
            val_dataset: pytorch validation dataset.
            train_step: pass train_function to customize training loop.
            eval_step: pass eval_function to customize training loop.
        """
        super(Trainer, self).__init__()
        optimizer = Adam or optimizer

        # logs are anything that can be pickled safely (non_objects)
        logs = DotDict(step=None,
                       epoch=None,
                       epochs=None,
                       num_batches=None,
                       batch_size=None,
                       lr=None,
                       momentum=None,
                       metrics=dict(),
                       train_out=None,  # output of train_step
                       val_out=None,  # output of val_Step
                       use_fp16=False)

        # objects in core must be saved in a special way
        core = DotDict(model=model,
                       optimizer=optimizer(model.parameters(), lr=3e-4),
                       criterion=criterion,
                       device='cuda' if torch.cuda.is_available() else 'cpu')

        self.state = DotDict(logs=logs, core=core)  # everything gets passed to callbacks

        cudnn.benchmark = True
        self.train_step = train_step or self._default_train_step
        self.eval_step = eval_step or self._default_eval_step
        self.callbacks = []

    @staticmethod
    def _default_train_step(batch, core):
        inputs, targets = batch
        inputs = inputs.to(core.device, non_blocking=True)
        targets = targets.to(core.device, non_blocking=True)
        outputs = core.model(inputs)
        loss = core.criterion(outputs, targets)
        return dict(loss=loss,
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    # by default, eval step is just the training step with `no_grad`
    @staticmethod
    @torch.no_grad()
    def _default_eval_step(batch, core):
        inputs, targets = batch
        inputs = inputs.to(core.device, non_blocking=True)
        targets = targets.to(core.device, non_blocking=True)
        outputs = core.model(inputs)
        loss = core.criterion(outputs, targets)
        return dict(loss=loss,
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    def fit(self, train_dataset, val_dataset=None, epochs=1, batch_size=32, lr=3e-4,
            num_workers=4, device_ids=None, callbacks=None):
        core = self.state.core
        logs = self.state.logs

        default_callbacks = [ProgressBar(), Metric(core.criterion, 'loss')]
        callbacks = Compose(default_callbacks + callbacks if callbacks else default_callbacks)

        if device_ids:
            core.model = torch.nn.DataParallel(core.model, device_ids)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)

        adjust_lr(core.optimizer, lr)
        logs.step = 0
        logs.epoch = 0
        logs.epochs = epochs
        logs.batch_size = batch_size
        logs.num_batches = len(train_loader)

        # try except here lets us break training within a callback by raising EndTraining error.
        try:
            callbacks.on_train_begin(self.state)

            for epoch in range(epochs):
                core.model.train()
                callbacks.on_epoch_begin(self.state)

                for batch in train_loader:
                    callbacks.on_batch_begin(self.state)

                    core.optimizer.zero_grad()
                    train_out = self.train_step(batch, core)

                    if logs.use_fp16:
                        try:
                            from apex import amp
                            with amp.scale_loss(train_out['loss'], core.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        except ImportError:
                            pass
                    else:
                        train_out['loss'].backward()

                    core.optimizer.step()
                    logs.step += 1
                    logs.train_out = train_out
                    logs.lr = core.optimizer.param_groups[0]['lr']
                    if core.optimizer.param_groups[0].get('momentum'):
                        logs.momentum = core.optimizer.param_groups[0]['momentum']

                    callbacks.on_batch_end(self.state)

                if val_dataset:
                    self.evaluate(val_dataset, batch_size, num_workers=4, device_ids=None, callbacks=callbacks)

                callbacks.on_epoch_end(self.state)
                logs.epoch += 1

            callbacks.on_train_end(self.state)
            self.callbacks = callbacks.callbacks
        except EndTraining as e:
            pass

    def evaluate(self, dataset, batch_size=32, num_workers=4, device_ids=None, callbacks=None):
        core = self.state.core
        logs = self.state.logs
        if callbacks and not isinstance(callbacks, Compose):
            callbacks = Compose([ProgressBar(), Metric(core.criterion, 'loss')] + callbacks)
        elif not callbacks:
            callbacks = Compose([ProgressBar(), Metric(core.criterion, 'loss')])
        else:
            callbacks = callbacks

        if device_ids:
            core.model = torch.nn.DataParallel(core.model, device_ids)
        core.model.eval()
        val_loader = DataLoader(dataset, batch_size, shuffle=False,
                                pin_memory=True, num_workers=num_workers)
        for batch in val_loader:
            callbacks.on_eval_batch_begin(self.state)
            val_out = self.eval_step(batch, core)
            logs.val_out = val_out

            callbacks.on_eval_batch_end(self.state)

        callbacks.on_eval_end(self.state)
        self.callbacks = callbacks.callbacks

    def find_lr(self, dataset, min_lr=1e-7, max_lr=1, batch_size=32, num_workers=4, smoothing=0.98):
        """calls fit with LRFinder callback
        Args:
            dataset:
            min_lr:
            max_lr:
            batch_size:
            num_workers:
            smoothing:
        """
        callbacks = [LRFinder(min_lr=min_lr, max_lr=max_lr, smoothing=smoothing)]
        self.fit(dataset, epochs=1, batch_size=batch_size, num_workers=num_workers, callbacks=callbacks)

    @torch.no_grad()
    def predict(self, batch):
        core = self.state.core
        core.model.eval()
        return core.model(batch.to(core.device))

    def save(self, path):
        core = self.state.core
        logs = self.state.logs
        checkpoint = dict(optimizer_state=core.optimizer.state_dict(),
                          model_state=core.model.state_dict(),
                          logs=logs)
        torch.save(checkpoint, path)

    def load(self, path):
        core = self.state.core
        checkpoint = torch.load(path)
        core.model.load_state_dict(checkpoint['model_state'])
        core.optimizer.load_state_dict(checkpoint['optimizer_state'])
        core.state = checkpoint['trainer_state']

    def save_model_trace(self, path, example_inputs=None):
        core = self.state.core
        logs = self.state.logs
        example_inputs = example_inputs if example_inputs else logs.train_out['inputs']
        trace = torch.jit.trace(core.model, example_inputs)
        trace.save(path)

    def to_fp16(self):
        core = self.state.core
        logs = self.state.logs
        amp_available = False
        try:
            from apex import amp
        except ImportError as e:
            amp_available = True
            warnings.warn(f"Error '{e}'' during importing apex library. To use mixed precison"
                          " you should install it from https://github.com/NVIDIA/apex")
        if amp_available:
            core.model, core.optimizer = amp.initialize(core.model, core.optimizer,
                                                        opt_level="O1", verbosity=0)
            logs.use_fp16 = True
        else:
            pass
