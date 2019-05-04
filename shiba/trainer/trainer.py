import copy
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from shiba.callbacks import Metric, ProgressBar
from shiba.trainer.base import Observable
from shiba.utils import adjust_lr
from shiba.vis import plot_lr_find


class Trainer(Observable):
    def __init__(self, model, criterion, optimizer, train_dataset, val_dataset=None, train_step=None, eval_step=None):
        super(Trainer, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_step = train_step if train_step else self._default_train_step
        self.eval_step = eval_step if eval_step else self._default_eval_step
        self.state = dict(step=0, epoch=0, device=self.device, metrics={})

    def _default_train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return dict(loss=loss.item(),
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    @torch.no_grad()
    def _default_eval_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return dict(loss=loss.item(),
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets)

    def fit(self, max_epochs=1, batch_size=32, lr=3e-4,
            num_workers=4, callbacks=None, metrics=None):

        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True,
                                  pin_memory=True, num_workers=num_workers)

        adjust_lr(self.optimizer, lr)
        default_callbacks = [ProgressBar(), Metric(self.criterion, 'loss')]
        self.register_callbacks(default_callbacks)

        if callbacks:
            self.register_callbacks(callbacks)
        if metrics:
            self.register_callbacks(metrics)

        self.state['max_epochs'] = max_epochs
        self.state['batch_size'] = batch_size
        self.state['len_loader'] = len(train_loader)

        self.train_begin()

        for epoch in range(max_epochs):

            self.epoch_begin()

            for batch in train_loader:
                self.batch_begin()
                train_output = self.train_step(batch)
                self.state['step'] += 1
                self.state['train_output'] = train_output
                self.state['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.batch_end()

            self.state['epoch'] += 1

            if self.val_dataset:
                val_loader = DataLoader(self.val_dataset, batch_size, shuffle=False,
                                        pin_memory=True, num_workers=num_workers)
                for batch in val_loader:
                    self.eval_batch_begin()
                    val_output = self.eval_step(batch)
                    self.state['val_output'] = val_output

                    self.eval_batch_end()

                self.eval_end()

            self.epoch_end()

        self.train_end()

    def find_lr(self, min_lr=1e-7, max_lr=1, batch_size=32, num_workers=4, smoothing=0.98):

        # checkpoint states before lr finder explodes them.
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        train_loader = DataLoader(self.train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        num_batches = len(train_loader) - 1
        mult = (max_lr / min_lr) ** (1 / num_batches)
        lr = min_lr
        avg_loss = 0
        best_loss = 0
        batch_num = 0
        losses = []
        lrs = []
        for batch in tqdm(train_loader):
            batch_num += 1
            train_output = self.train_step(batch)
            # Compute the smoothed loss
            avg_loss = smoothing * avg_loss + (1 - smoothing) * train_output['loss']
            smoothed_loss = avg_loss / (1 - smoothing ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            losses.append(smoothed_loss)
            lrs.append(lr)
            lr *= mult
            adjust_lr(self.optimizer, lr)
        plot_lr_find(lrs, losses)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        return self.model(batch.to(self.device))
