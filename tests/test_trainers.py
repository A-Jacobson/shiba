import torch
from shiba import Trainer
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

model = resnet18()
model.fc = nn.Linear(512, 10)

train_dataset = CIFAR10('data', train=True, download=True, transform=ToTensor())
val_dataset = CIFAR10('data', train=False, download=True, transform=ToTensor())

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, criterion, optimizer, train_dataset, val_dataset)

trainer.fit(max_epochs=1)
