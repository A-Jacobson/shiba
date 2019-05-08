import torch
from shiba.callbacks import Compose
from shiba.callbacks import Metric, ProgressBar, LRFinder
from shiba.utils import adjust_lr
from torch.optim import Adam
from torch.utils.data import DataLoader


class Trainer:
    """Shiba Trainer"""

    def __init__(self, model, criterion, train_dataset, val_dataset=None,
                 callbacks=None, optimizer=None, train_step=None, eval_step=None):
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
        self.criterion = criterion
        self.model = model
        optimizer = Adam or optimizer
        self.optimizer = optimizer(model.parameters())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_step = train_step or self._default_train_step
        self.eval_step = eval_step or self._default_eval_step
        self.state = dict(step=0,
                          epoch=0,
                          device=self.device,
                          train_metrics=dict(),
                          val_metrics=dict())
        self.default_callbacks = [ProgressBar(), Metric(self.criterion, 'loss')]
        callbacks = self.default_callbacks + callbacks if callbacks else self.default_callbacks
        self.callbacks = Compose(callbacks)

    def _default_train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item(),
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    # meow
    @torch.no_grad()
    def _default_eval_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return dict(loss=loss.item(),
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    def fit(self, max_epochs=1, batch_size=32, lr=3e-4,
            num_workers=4):

        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)

        adjust_lr(self.optimizer, lr)

        self.state['max_epochs'] = max_epochs
        self.state['batch_size'] = batch_size
        self.state['num_batches'] = len(train_loader)

        self.callbacks.on_train_begin(self.state)

        for epoch in range(max_epochs):

            self.callbacks.on_epoch_begin(self.state)

            for batch in train_loader:
                self.callbacks.on_batch_begin(self.state)

                train_output = self.train_step(batch)
                self.state['step'] += 1
                self.state['train_output'] = train_output
                self.state['learning_rate'] = self.optimizer.param_groups[0]['lr']

                self.callbacks.on_batch_end(self.state)

            self.state['epoch'] += 1

            if self.val_dataset:
                val_loader = DataLoader(self.val_dataset, batch_size, shuffle=False,
                                        pin_memory=True, num_workers=num_workers)
                for batch in val_loader:
                    self.callbacks.on_eval_batch_begin(self.state)
                    val_output = self.eval_step(batch)
                    self.state['val_output'] = val_output

                    self.callbacks.on_eval_batch_end(self.state)

                self.callbacks.on_eval_end(self.state)

            self.callbacks.on_epoch_end(self.state)

        self.callbacks.on_train_end(self.state)

    def find_lr(self, min_lr=1e-7, max_lr=1, batch_size=32, num_workers=4, smoothing=0.98):
        # save original callbacks, hide val dataset to skip validation
        callbacks_tmp = self.callbacks
        val_dataset_tmp = self.val_dataset
        self.val_dataset = None
        lr_finder = LRFinder(model=self.model, optimizer=self.optimizer,
                             min_lr=min_lr, max_lr=max_lr,
                             smoothing=smoothing)
        self.callbacks = Compose(self.default_callbacks + [lr_finder])
        self.fit(max_epochs=1, batch_size=batch_size, num_workers=num_workers)
        self.callbacks = callbacks_tmp
        self.val_dataset = val_dataset_tmp

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        return self.model(batch.to(self.device))
