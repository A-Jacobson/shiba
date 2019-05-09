# shiba
A simple, flexible, pytorch training loop in the spirit of keras. There are many like it, but this one is mine.

### Features
- callbacks
- learning rate finder
- learning rate schedulers
- tensorboard and comet visualizations
- one_cycle
- multi-gpu training
- mixed precision training (using apex)
- checkpointer
- early stopping (todo)
- confusion matrix + comet/tb visualizations (todo)
- (maybe) grpc/rest deployment hook

## Install
```bash
pip install git+https://github.com/A-Jacobson/shiba.git
```

### Train resnet18 on CIFAR10 with tensorboard logging
```python
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18

from shiba import Trainer
from shiba.callbacks import TensorBoard, Save, Metric

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('data', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10('data', train=False, download=True, transform=ToTensor())

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
callbacks = [TensorBoard(),
             Save('weights', interval=2)
            ]   
            
# optimizer defaults to Adam
trainer = Trainer(model, criterion, train_dataset, val_dataset, callbacks) 

#lr defaults to 3e-4 (the best learning rate https://twitter.com/karpathy/status/801621764144971776?lang=en) 
trainer.fit(epochs=10)
```

### Write your own training steps and validation steps.
shiba comes with sensible default steps that can be easily overriden by passing your own
 `train_step` and `val_step` functions to the constructor. 
```python
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
                

def custom_step(batch, core):
    """
    core contains : model, optimizer, criterion, datasets, and device
    """
    # training stuff here!
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                targets=targets)
                

trainer = Trainer(model, criterion, train_dataset, train_step=custom_step)
```

### Use Callbacks to easily add support for logging, Progress bars, metrics, and learning rate schedulers.
```python
class ProgressBar(Callback):
    def __init__(self):
        self.train_pbar = None
        self.epoch_pbar = None

    def on_train_begin(self, state):
        epochs = state.logs.epochs
        self.train_pbar = tqdm(range(epochs), total=epochs, unit='epochs')

    def on_epoch_begin(self, state):
        self.epoch_pbar = tqdm(total=state.logs.num_batches, unit='b')

    def on_epoch_end(self, state):
        self.train_pbar.update()
        self.epoch_pbar.close()

    def on_batch_end(self, state):
        self.epoch_pbar.update()
        self.epoch_pbar.set_postfix(state.logs.metrics)

    def on_eval_end(self, state):
        self.epoch_pbar.set_postfix(state.logs.metrics)

    def on_train_end(self, state):
        self.train_pbar.close()

 ```