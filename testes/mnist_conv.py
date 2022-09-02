import torch
import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, log_softmax
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

## Example from pytorch
## https://github.com/pytorch/examples/blob/main/mnist/main.py

class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)

        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)

        return output


def mk_mnist_conv(traindt, validdt, testdt, gpu):
    epochs       = 10
    criterion    = nn.NLLLoss()
    optim_params = { 'lr' : 0.01}
    optimizer    = optim.Adadelta


    return ModelConfig(MnistConv(),
                       criterion,
                       optimizer,
                       optim_params,
                       traindt,
                       validdt,
                       testdt,
                       epochs=epochs,
                       gpu=gpu)

if __name__ == '__main__':
    model = MnistConv()
    summary(model)
