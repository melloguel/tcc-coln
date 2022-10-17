import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch import optim
from torchinfo import summary

from coln import AbstractModel
from modelconfig import ModelConfig

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32*32*3, 10000)
        self.linear2 = nn.Linear(10000, 10)

    def forward(self, image):
        a = image.view(-1, 32*32*3)
        a = relu(self.linear1(a))
        a = relu(self.linear2(a))
        output = softmax(a, dim=1)
        return output


def mk_cifar10_smlp(traindt, validdt, testdt, device):
    epochs       = 8
    criterion    = nn.CrossEntropyLoss()
    optim_params = {'lr': 0.01, 'momentum': 0.9}
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
