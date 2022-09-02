from torch import nn
from torch.nn.functional import relu
from torch.nn.functional import log_softmax
from torch import tanh
from torch import optim
from torchinfo import summary
from modelconfig import ModelConfig

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 160)
        self.fc3 = nn.Linear(160, 120)
        self.fc4 = nn.Linear(120, 80)
        self.fc5 = nn.Linear(80, 50)
        self.fc6 = nn.Linear(50, 10)

        self.d1 = nn.Dropout(0.25)
        self.d2 = nn.Dropout(0.10)


    def forward(self, image):
        a = image.view(-1, 28*28)
        a = relu(self.fc1(a))
        a = relu(self.fc2(a))
        a = self.d1(a)

        a = relu(self.fc3(a))
        a = relu(self.fc4(a))
        a = self.d2(a)

        a = tanh(self.fc5(a))
        a = log_softmax(self.fc6(a), dim=1)
        return a


def mk_mnist_lmlp(traindt, validdt, testdt, gpu):
    epochs       = 20
    criterion    = nn.NLLLoss()
    optim_params = { 'lr' : 0.01, 'momentum': 0.9}
    optimizer    = optim.SGD

    return ModelConfig(LargeMLP(),
                       criterion,
                       optimizer,
                       optim_params,
                       traindt,
                       validdt,
                       testdt,
                       epochs=epochs,
                       gpu=gpu)

if __name__ == '__main__':
    model = LargeMLP()
    summary(model)
