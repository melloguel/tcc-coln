import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

## Example from pytorch
## https://github.com/pytorch/examples/blob/main/mnist/main.py

class MNISTConv(nn.Module):
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
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def mk_mnist_conv(traindt, validdt, testdt, device):
    epochs       = 14
    criterion = nn.NLLLoss(reduction='sum')
    optim_params = { 'lr' : 1.0}
    optimizer    = optim.Adadelta

    return ModelConfig(
            model=MNISTConv(),
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=optim_params,
            traindt=traindt,
            validdt=validdt,
            testdt=testdt,
            epochs=epochs,
            device=device)

if __name__ == '__main__':
    model = MnistConv()
    summary(model)
