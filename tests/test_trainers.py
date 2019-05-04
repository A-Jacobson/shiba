from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from shiba.callbacks import TensorBoard
from shiba.trainer import Trainer

transform = transforms.ToTensor()
dataset = CIFAR10('../cifar', train=True, download=True, transform=transform)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters())

trainer = Trainer(net, optimizer, criterion, dataset)

trainer.fit(dataset, callbacks=[TensorBoard()], cuda=True, nb_epoch=3)
