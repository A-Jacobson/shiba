from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from shiba import Trainer
from shiba.callbacks import TensorBoard, Metric, Save
from shiba.metrics import categorical_accuracy
from shiba.vis import classification_snapshot

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('cifar', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10('cifar', train=False, download=True, transform=ToTensor())
callbacks = [TensorBoard(snapshot_func=classification_snapshot),
             Metric(categorical_accuracy, 'accuracy'),
             Save('weights', interval=2)
             ]

criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, criterion, train_dataset, val_dataset, callbacks)
trainer.fit(epochs=1)
