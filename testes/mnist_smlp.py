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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        output = self.softmax(a)
        return output


def mk_mnist_smlp(traindt, validdt, testdt, device):
    epochs       = 15
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.01, 'momentum' : 0.9 }
    optimizer    = optim.SGD


    return ModelConfig(
            model=SimpleMLP(),
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=optim_params,
            traindt=traindt,
            validdt=validdt,
            testdt=testdt,
            epochs=epochs,
            device=device)

if __name__ == '__main__':
    model = SimpleMLP()
    summary(model)
