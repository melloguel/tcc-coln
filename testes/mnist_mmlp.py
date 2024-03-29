import torch
import torch.nn as nn
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

# Modelo MNIST 2NN de McMahan et al, 2017
class MMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.linear2 = nn.Linear(200, 200)
        self.final   = nn.Linear(200, 10)
        self.relu    = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.final(a)
        output = self.softmax(a)
        return output


def mk_mnist_mmlp(traindt, validdt, testdt, device):
    epochs       = 20
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.1, 'momentum' : 0.09 }
    optimizer    = optim.SGD
    scheduler    = optim.lr_scheduler.StepLR
    sched_params = { 'step_size' : 1, 'gamma' : 0.7 }

    return ModelConfig(
            model=MMLP(),
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=optim_params,
            scheduler=scheduler,
            scheduler_params=sched_params,
            traindt=traindt,
            validdt=validdt,
            testdt=testdt,
            epochs=epochs,
            device=device)

if __name__ == '__main__':
    model = MMLP()
    summary(model)
