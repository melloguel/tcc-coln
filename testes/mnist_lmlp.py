from torch import nn
from torch.nn.functional import relu
from torch import optim
from torchinfo import summary
from modelconfig import ModelConfig

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 90)
        self.fc3 = nn.Linear(90, 75)
        self.fc4 = nn.Linear(75, 60)
        self.fc5 = nn.Linear(60, 55)
        self.fc6 = nn.Linear(55, 50)
        self.fc7 = nn.Linear(50, 30)
        self.fc8 = nn.Linear(30, 10)


    def forward(self, image):
        a = image.view(-1, 28*28)
        a = relu(self.fc1(a))
        a = relu(self.fc2(a))
        a = relu(self.fc3(a))
        a = relu(self.fc4(a))
        a = relu(self.fc5(a))
        a = relu(self.fc6(a))
        a = relu(self.fc7(a))
        a = self.fc8(a)
        return a


def mk_mnist_lmlp(traindt, testdt):
    epochs       = 40
    criterion    = nn.CrossEntropyLoss()
    optim_params = { 'lr' : 0.01, 'momentum' : 0.9 }
    optimizer    = optim.SGD

    return ModelConfig(LargeMLP(),
                       criterion,
                       optimizer,
                       optim_params,
                       traindt,
                       testdt,
                       epochs=epochs,
                       gpu=True)

if __name__ == '__main__':
    model = LargeMLP()
    summary(model)
