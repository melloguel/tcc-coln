import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchinfo import summary

from modelconfig import ModelConfig

class MNISTRNN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.hidden_size     = 128
        self.input_size      = 28
        self.num_classes     = 10
        self.num_layers      = 2
        self.sequence_length = 28
        self.device = device
        
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, img):
        x  = img.reshape(-1, self.sequence_length, self.input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, hidden = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def mk_mnist_rnn(traindt, validdt, testdt, device):
    epochs       = 2
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.01 }
    optimizer    = optim.Adam
    scheduler    = optim.lr_scheduler.StepLR
    sched_params = { 'step_size' : 1, 'gamma' : 0.7 }

    return ModelConfig(
            model=MNISTRNN(device),
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
    model = MNISTRNN('cpu')
    summary(model)
