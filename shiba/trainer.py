import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from shiba.callbacks import Compose
from shiba.callbacks import Metric, ProgressBar, LRFinder, OneCycle
from shiba.steps import default_train_step, default_eval_step
from shiba.utils import adjust_lr, DotDict, EndTraining


class Trainer:
    """Shiba Trainer"""

    def __init__(self, model, criterion,
                 optimizer=None, train_step=None, eval_step=None):
        """
        Args:
            model: pytorch model.
            criterion: loss function.
            optimizer: pytorch optimizer. defaults to Adam
            train_step: pass train_function to customize training loop.
            eval_step: pass eval_function to customize training loop.
        """
        super(Trainer, self).__init__()
        optimizer = optimizer or Adam

        # logs are anything that can be pickled safely (non_objects)
        logs = DotDict(step=None,
                       epoch=None,
                       epochs=None,
                       global_step=0,
                       num_batches=None,
                       batch_size=None,
                       lr=None,
                       momentum=None,
                       metrics=dict())

        # objects in are passed to steps, are related to the model in some way
        core = DotDict(model=model,
                       optimizer=optimizer(model.parameters(), lr=3e-4),
                       criterion=criterion,
                       device='cuda' if torch.cuda.is_available() else 'cpu',
                       train_out=dict(),  # output of previous train_step
                       val_out=dict(),  # output of previous val_Step
                       use_fp16=False)

        self.state = DotDict(logs=logs, core=core)  # everything gets passed to callbacks

        cudnn.benchmark = True
        self.train_step = train_step or default_train_step
        self.eval_step = eval_step or default_eval_step
        self.callbacks = []

    def fit(self, train_dataset, val_dataset=None, epochs=1, lr=3e-4,
            batch_size=32, num_workers=4, device_ids=None, callbacks=None):
        """
        Args:
            train_dataset: Pytorch Dataset or loader
            val_dataset: Pytorch Dataset or loader
            epochs: num_epochs to train
            lr: learning rate
            batch_size: sets, batch size on loader ignored if loader is passed
            num_workers: num_workers for data loader, ignored if loader is passed
            device_ids: [0, 1, 2] gpu device ids for multi-gpu training
            callbacks: list of callbacks
        Returns:

        """
        core = self.state.core
        logs = self.state.logs

        default_callbacks = [ProgressBar(), Metric('loss', transform=lambda x: x['loss'].item())]
        callbacks = Compose(default_callbacks + callbacks if callbacks else default_callbacks)

        if device_ids:
            core.model = torch.nn.DataParallel(core.model, device_ids)

        if isinstance(train_dataset, Dataset):
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                                      pin_memory=True, num_workers=num_workers)
        else:
            train_loader = train_dataset

        adjust_lr(core.optimizer, lr)
        logs.step = 0
        logs.epoch = 0
        logs.epochs = epochs
        logs.batch_size = train_loader.batch_size
        logs.num_batches = len(train_loader)

        # try except here lets us break training within a callback by raising EndTraining error.
        try:
            callbacks.on_train_begin(self.state)

            for epoch in range(epochs):
                core.model.train()
                # cache hidden state for sequence models, core is passed to step_function
                if hasattr(core.model, 'init_hidden'):
                    core.train_out['hidden'] = core.model.init_hidden(logs.batch_size)
                    if hasattr(train_loader, 'seq_len'):
                        core.seq_len = train_loader.seq_len  # TODO this relies on seq_len being set on the loader

                callbacks.on_epoch_begin(self.state)

                for batch in train_loader:
                    callbacks.on_batch_begin(self.state)

                    core.optimizer.zero_grad()
                    train_out = self.train_step(batch, core)
                    core.train_out = train_out  # save training output to core state
                    if core.use_fp16:
                        try:
                            from apex import amp
                            with amp.scale_loss(train_out['loss'], core.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        except ImportError:
                            pass
                    else:
                        train_out['loss'].backward()

                    logs.lr = core.optimizer.param_groups[0]['lr']
                    if core.optimizer.param_groups[0].get('momentum'):
                        logs.momentum = core.optimizer.param_groups[0]['momentum']

                    callbacks.on_batch_end(self.state)
                    core.optimizer.step()
                    logs.step += 1
                    logs.global_step += 1

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
        if callbacks and not isinstance(callbacks, Compose):
            callbacks = Compose([ProgressBar(), Metric('loss', transform=lambda x: x['loss'].item())] + callbacks)
        elif not callbacks:
            callbacks = Compose([ProgressBar(), Metric('loss', transform=lambda x: x['loss'].item())])
        else:
            callbacks = callbacks

        if isinstance(dataset, Dataset):
            val_loader = DataLoader(dataset, batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers)
        else:
            val_loader = dataset

        if device_ids:
            core.model = torch.nn.DataParallel(core.model, device_ids)
        core.model.eval()
        # cache hidden state for sequence models, core is passed to step_function
        if hasattr(core.model, 'init_hidden'):
            core.train_out['hidden'] = core.model.init_hidden(val_loader.batch_size)
            if hasattr(val_loader, 'seq_len'):
                core.seq_len = val_loader.seq_len  # TODO this relies on seq_len being set on the loader

        for batch in val_loader:
            callbacks.on_eval_batch_begin(self.state)

            core.val_out = self.eval_step(batch, core)

            callbacks.on_eval_batch_end(self.state)

        callbacks.on_eval_end(self.state)
        self.callbacks = callbacks.callbacks

    def find_lr(self, dataset, min_lr=1e-7, max_lr=1, batch_size=32, num_workers=4):
        """calls fit with LRFinder callback
        Args:
            dataset:
            min_lr:
            max_lr:
            batch_size:
            num_workers:
        """
        callbacks = [LRFinder(min_lr=min_lr, max_lr=max_lr)]
        self.fit(dataset, epochs=1, batch_size=batch_size, num_workers=num_workers, callbacks=callbacks)

    def fit_one_cycle(self, train_dataset, val_dataset=None, epochs=1, batch_size=32, max_lr=1e-3,
                      end_percentage=0.1, scale_percentage=None, maximum_momentum=0.95, minimum_momentum=0.85,
                      num_workers=4, device_ids=None, callbacks=None):
        one_cycle = OneCycle(max_lr=max_lr,
                             end_percentage=end_percentage,
                             scale_percentage=scale_percentage,
                             maximum_momentum=maximum_momentum,
                             minimum_momentum=minimum_momentum)
        if callbacks:
            callbacks = callbacks + [one_cycle]
        else:
            callbacks = [one_cycle]
        self.fit(train_dataset, val_dataset, epochs=epochs,
                 batch_size=batch_size, num_workers=num_workers,
                 device_ids=device_ids, callbacks=callbacks)

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

    def to_fp16(self, opt_level="01"):
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
                                                        opt_level=opt_level, verbosity=0)
            logs.use_fp16 = True
        else:
            pass
