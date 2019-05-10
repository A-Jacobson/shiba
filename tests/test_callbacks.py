from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from shiba import Trainer
from shiba.callbacks import TensorBoard, Metric, Save
from shiba.metrics import categorical_accuracy
from shiba.vis import vis_classify

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('data/cifar', train=True,  transform=ToTensor())
val_dataset = CIFAR10('data/cifar', train=False, transform=ToTensor())
callbacks = [TensorBoard(vis_function=vis_classify),
             Metric(categorical_accuracy, 'accuracy'),
             Save('weights', interval=2)
             ]

criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, criterion)
trainer.fit(train_dataset, val_dataset, epochs=1, callbacks=callbacks)


