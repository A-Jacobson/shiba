# shiba
A simple, flexible, pytorch trainer. Lighter (just a trainer) and lower level than `fastai`, higher level than `ignite`.
There are many like it, but I've tried them all and wasn't satisfied.
So, I'm shamelessly stealing the best (my opinion) parts of each.

### Features 
- `callbacks/fit api` (keras/sklearn)
- `learning rate finder` (fastai)
- `one_cycle` (fastai)
- `mixed precision training` (apex)
- `process_functions/process_function zoo` (ignite/me)
- `output_transforms for metrics` (ignite)
- `tensorboard prediction vis functions` (me)

## Install
```bash
pip install git+https://github.com/A-Jacobson/shiba.git
```

### Train resnet18 on CIFAR10 with tensorboard logging, Checkpointing, and a customer Metric.
```python
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18

from shiba import Trainer
from shiba.callbacks import TensorBoard, Save, Metric
from shiba.vis import vis_classify
from shiba.metrics import categorical_accuracy

train_dataset = CIFAR10('data', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10('data', train=False, transform=ToTensor())

model = resnet18()
model.fc = nn.Linear(512, 10)
criterion = nn.CrossEntropyLoss()      
trainer = Trainer(model, criterion) 
trainer.find_lr(train_dataset) # prints lr finder graph

callbacks = [TensorBoard(log_dir='runs/shiba-test-cifar', vis_function=vis_classify),
             Metric(name='accuracy', score_func=categorical_accuracy),
             Save('weights/cifar', monitor='val_loss')]

trainer.fit_one_cycle(train_dataset, val_dataset, epochs=10, max_lr=1e-3, callbacks=callbacks)
```

### Write your own training steps and validation steps.
shiba comes with sensible default steps that can be easily overridden by passing your own
 `train_step` and/or `val_step` functions to the constructor. 
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
                

trainer = Trainer(model, criterion, train_step=custom_step)
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