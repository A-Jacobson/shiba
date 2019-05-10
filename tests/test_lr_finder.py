from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from shiba import Trainer

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('data/cifar', train=True,  transform=ToTensor())

criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, criterion, train_dataset)
trainer.find_lr(train_dataset)
