from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from shiba import Trainer


def test_trainer():
    model = resnet18()
    model.fc = nn.Linear(512, 10)

    train_dataset = CIFAR10('cifar', train=True, download=True, transform=ToTensor())
    val_dataset = CIFAR10('cifar', train=False, download=True, transform=ToTensor())

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, criterion, train_dataset, val_dataset)

    trainer.fit(epochs=1)
    assert trainer.state.logs.train_metrics['train_loss']
