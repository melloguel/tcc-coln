import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

class LMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(30, 256)
        self.linear2 = nn.Linear(256,128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.final   = nn.Linear(32, 2)

    def _initiliaze_layers(self):
        nn.init.uniform(self.linear1.weight, a=-0.05, b=0.05)
        nn.init.uniform(self.linear2.weight, a=-0.05, b=0.05)
        nn.init.uniform(self.linear3.weight, a=-0.05, b=0.05)
        nn.init.uniform(self.linear4.weight, a=-0.05, b=0.05)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        a = F.relu(self.linear1(x))
        a = F.dropout(a, 0.1)

        a = F.relu(self.linear2(a))

        a = F.relu(self.linear3(a))

        a = F.relu(self.linear4(a))
        a = self.final(a)

        output = F.softmax(a, dim=1)
        return output


def mk_wisconsin_lmlp(traindt, validdt, testdt, device):
    epochs       = 200
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.01 }
    optimizer    = optim.Adam
    scheduler    = optim.lr_scheduler.StepLR
    sched_params = { 'step_size' : 1, 'gamma' : 0.7 }


    return ModelConfig(
            model=LMLP(),
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
    model = LMLP()
    summary(model, (1, 30))
