import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, softmax
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

class SMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32*32*3, 2000)
        self.linear2 = nn.Linear(2000, 5000)
        self.linear3 = nn.Linear(5000, 10)

    def forward(self, image):
        a = image.view(-1, 32*32*3)
        a = relu(self.linear1(a))
        a = F.dropout(a, 0.3)
        a = relu(self.linear2(a))
        a = F.dropout(a, 0.4)
        a = relu(self.linear3(a))
        output = softmax(a, dim=1)
        return output


def mk_cifar10_smlp(traindt, validdt, testdt, device):
    epochs       = 16
    criterion    = nn.CrossEntropyLoss()
    optim_params = {'lr': 0.01, 'momentum': 0.9}
    optimizer    = optim.SGD
    scheduler    = optim.lr_scheduler.StepLR
    sched_params = { 'step_size' : 1, 'gamma' : 0.7 }

    return ModelConfig(
            model=SMLP(),
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
    model = SMLP()
    summary(model)
