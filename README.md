# shiba
A simple, flexible, pytorch training loop in the spirit of keras.

### Train resnet18 on CIFAR10 with tensorboard logging
```python
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18

from shiba import Trainer
from shiba.callbacks import TensorBoard

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('data', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10('data', train=False, download=True, transform=ToTensor())

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, criterion, optimizer, train_dataset, val_dataset)

trainer.fit(max_epochs=10, callbacks=[TensorBoard()])
```

### Write your own training steps and validation steps.
shiba comes with sensible default steps that can be easily overriden by passing your own
 `train_step` and `val_step` functions to the constructor. 
```python
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
                

def custom_step(self, batch):
    # training stuff here!
    return dict(loss=loss.item(),
                inputs=inputs,
                outputs=outputs,
                targets=targets)
                

trainer = Trainer(model, criterion, optimizer, train_dataset, train_step=custom_step)
```

### Use Callbacks to easily add support for logging, Progress bars, metrics, and learning rate schedulers.
```python
class TensorBoard(Callback):
    def __init__(self, experiment_name='', snapshot_func=None, hyperparams=None):
        self.snapshot_func = snapshot_func
        self.hyperparams = hyperparams
        self.writer = SummaryWriter(comment=experiment_name)

    def on_train_begin(self, state):
        if self.hyperparams:
            text = ""
            for name, value in self.hyperparams.items():
                text += f'{name}: {str(value)}  '
            self.writer.add_text('hyperparams', text, state.get('epoch'))

    def on_epoch_end(self, state):
        epoch = state.get('epoch')
        for metric, value in state['metrics'].items():
            self.writer.add_scalar(metric, value, epoch)

        if self.snapshot_func:
            epoch = state['epoch']
            val_output = state['val_output']
            inputs = val_output['inputs']
            outputs = val_output['outputs']
            targets = val_output['targets']
            snapshot = self.snapshot_func(inputs, outputs, targets)
            for name, value in snapshot.items():
                self.writer.add_image(name, value, epoch)
 ```