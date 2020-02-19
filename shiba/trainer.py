import warnings

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam

from shiba.callbacks import Compose
from shiba.callbacks import Metric, ProgressBar, LRFinder, OneCycle
from shiba.steps import default_step
from shiba.utils import adjust_lr, EndTraining, model_to_devices


class Trainer:
    """Shiba Trainer"""
    def __init__(self, model, criterion, optimizer=None, train_step=None, eval_step=None):
        """
        Args:
            model: pytorch model.
            criterion: loss function
            optimizer: pytorch optimizer. defaults to Adam
            train_step: pass train_function to customize training loop.
            eval_step: pass eval_function to customize training loop.
        """
        self.model = model
        self.optimizer = self._build_optimizer(optimizer)
        self.criterion = criterion

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_apex = False

        self.out = dict()
        self.metrics = dict()

        self.step = None
        self.epoch = None
        self.epochs = None
        self.global_step = 0
        self.num_batches = None
        self.num_val_batches = None
        self.batch_size = None

        self.callbacks = []

        self.train_step = train_step or default_step
        self.eval_step = eval_step or self.train_step

    def fit(self, train_loader, val_loader=None, epochs=1, lr=3e-4, callbacks=None, device_ids=-1):
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
        cudnn.benchmark = True
        self.step = 0
        self.epoch = 0
        self.epochs = epochs
        adjust_lr(self.optimizer, lr)
        default_callbacks = [ProgressBar(),
                             Metric('loss', transform=lambda x: x['loss'].item())]
        callbacks = self._set_callbacks(callbacks, default_callbacks)
        self.model = model_to_devices(self.model, self.device, device_ids)
        self.num_batches = len(train_loader)
        # try except here lets us break training within a callback by raising EndTraining error.
        try:
            callbacks.on_train_begin(self)
            for epoch in range(epochs):
                self.batch_size = train_loader.batch_size  # HACK, set like this to cover LMloader.
                self.model.train()
                # cache hidden trainer for sequence models, core is passed to step_function
                self.out['hidden'] = self.init_hidden()
                callbacks.on_epoch_begin(self)
                for batch in train_loader:
                    callbacks.on_batch_begin(self)
                    self.optimizer.zero_grad()
                    self.out = self.train_step(self, batch)
                    self.backward()
                    callbacks.on_batch_end(self)
                    self.optimizer.step()
                    self.step += 1
                    self.global_step += 1
                if val_loader:
                    self.evaluate(val_loader, callbacks=callbacks)
                callbacks.on_epoch_end(self)
                self.epoch += 1
            callbacks.on_train_end(self)
            self.callbacks = callbacks.callbacks
        except EndTraining as e:
            pass

    @torch.no_grad()
    def evaluate(self, data_loader, callbacks=None, device_ids=None, pbar=True):
        default_callbacks = [ProgressBar(val_bar=pbar),
                             Metric('loss', transform=lambda x: x['loss'].item())]
        callbacks = self._set_callbacks(callbacks, default_callbacks)
        self.batch_size = data_loader.batch_size  # HACK, set like this to cover LMloader.
        self.model = model_to_devices(self.model, self.device, device_ids)
        self.model.eval()
        self.out['hidden'] = self.init_hidden()  # check if rnn and cache hidden trainer for sequence models
        self.num_val_batches = len(data_loader)
        callbacks.on_eval_begin(self)
        for batch in data_loader:
            callbacks.on_eval_batch_begin(self)
            self.out = self.eval_step(self, batch)
            callbacks.on_eval_batch_end(self)
        callbacks.on_eval_end(self)
        self.callbacks = callbacks.callbacks

    def find_lr(self, data_loader, min_lr=1e-7, max_lr=1):
        """calls fit with LRFinder callback
        Args:
            dataset:
            min_lr:
            max_lr:
            batch_size:
            num_workers:
        """
        callbacks = [LRFinder(min_lr=min_lr, max_lr=max_lr)]
        self.fit(data_loader, epochs=1, callbacks=callbacks)

    def fit_one_cycle(self, train_loader, val_loader=None, epochs=1, max_lr=1e-3, callbacks=None,
                      end_percentage=0.1, scale_percentage=None, maximum_momentum=0.95, minimum_momentum=0.85,
                      device_ids=None):
        one_cycle = OneCycle(max_lr=max_lr,
                             end_percentage=end_percentage,
                             scale_percentage=scale_percentage,
                             maximum_momentum=maximum_momentum,
                             minimum_momentum=minimum_momentum)
        callbacks = callbacks + [one_cycle] if callbacks else [one_cycle]
        self.fit(train_loader, val_loader, epochs=epochs, callbacks=callbacks, device_ids=device_ids)

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        return self.model(batch.to(self.device))

    def save(self, path):
        checkpoint = dict(optimizer_state=self.optimizer.state_dict(),
                          model_state=self.model.state_dict(),
                          metrics=self.metrics,
                          global_step=self.global_step)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.metrics = checkpoint['metrics']
        self.global_step = checkpoint['global_step']

    def save_model_trace(self, path, example_inputs=None):
        example_inputs = example_inputs if isinstance(example_inputs, torch.Tensor) else self.out['inputs']
        trace = torch.jit.trace(self.model, example_inputs)
        trace.save(path)

    def precision(self, level="fp32"):
        """
        Args:
            level: "fp32, Mixed_S, Mixed_F, fp16"
        """
        levels = {'fp32': 'O0', 'Mixed_S': 'O1', 'Mixed_F': 'O2', 'fp16': 'O3'}
        if torch.cuda.is_available():
            try:
                from apex import amp
                self.model, self.optimizer = amp.initialize(self.model.to('cuda'), self.optimizer,
                                                            opt_level=levels[level], verbosity=0)
                self.use_apex = True
                print(f'Amp enabled level: {level}')
                print('see https://nvidia.github.io/apex/amp.html for more details.')
            except ImportError as e:
                warnings.warn(f"Error '{e}'' during importing apex library. To use mixed precision"
                              " you should install it by running  "
                              'pip install git+https://github.com/NVIDIA/apex.git -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"')
        else:
            print('fp16 training only available on gpu.')

    def _build_optimizer(self, optimizer):
        optimizer = optimizer or Adam
        return optimizer(self.model.parameters(), lr=3e-4)

    def backward(self):
        """backward pass, optionally with apex loss scaling"""
        if self.use_apex:
            try:
                from apex import amp
                with amp.scale_loss(self.out['loss'], self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                pass
        else:
            self.out['loss'].backward()

    @staticmethod
    def _set_callbacks(callbacks, default_callbacks):
        if callbacks and not isinstance(callbacks, Compose):
            return Compose(default_callbacks + callbacks)
        elif not callbacks:
            return Compose(default_callbacks)
        else:
            return callbacks

    def init_hidden(self):
        hidden = None
        if hasattr(self.model, 'init_hidden'):
            hidden = self.model.init_hidden(self.batch_size)
        return hidden
