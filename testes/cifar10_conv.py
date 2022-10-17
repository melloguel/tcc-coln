import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

## Base Example from
## https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

class CIFAR10Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 2, padding=1)

        self.pool  = nn.MaxPool2d(2)

        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 64)
        self.fc4   = nn.Linear(64, 10)

        self._weights_initilization()

    def _weights_initilization(self):
        nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv2.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv3.weight, a=-0.05, b=0.05)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

def mk_cifar10_conv(traindt, validdt, testdt, device):
    epochs       = 12
    criterion = nn.NLLLoss(reduction='sum')
    optim_params = {}
    optimizer    = optim.Adadelta

    return ModelConfig(
            model=CIFAR10Conv(),
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=optim_params,
            traindt=traindt,
            validdt=validdt,
            testdt=testdt,
            epochs=epochs,
            device=device)

if __name__ == '__main__':
    model = CIFAR10Conv()
    summary(model)
