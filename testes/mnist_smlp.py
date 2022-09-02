import torch
import torch.nn as nn
from torch import optim
from torchinfo import summary

from coln import AbstractModel
from modelconfig import ModelConfig

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final   = nn.Linear(50, 10)
        self.relu    = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        return a


def mk_mnist_smlp(traindt, validdt, testdt, gpu):
    epochs       = 15
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.01, 'momentum' : 0.9 }
    optimizer    = optim.SGD


    return ModelConfig(SimpleMLP(),
                       criterion,
                       optimizer,
                       optim_params,
                       traindt,
                       validdt,
                       testdt,
                       epochs=epochs,
                       gpu=gpu)

if __name__ == '__main__':
    model = SimpleMLP()
    summary(model)
