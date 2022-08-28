# coding: utf-8
'''
Setup to train classifiers.
'''
import torch
from coln import AbstractModel

class ModelConfig(AbstractModel):
    def __init__(self, model, criterion, optimizer, optimizer_params,
                 traindt, testdt, epochs=10, gpu=None):
        self.model   = model
        self.criter  = criterion
        self.optim   = optimizer(model.parameters(), **optimizer_params)
        self.traindt = traindt
        self.testdt  = testdt
        self.epochs  = epochs
        self.gpu     = gpu

    def get_layers(self):
        self.model.train(mode=False)
        return list(self.model.parameters())

    def set_layers(self, new_layers):
        old_layers = self.get_layers()
        with torch.no_grad():
            for old, new in zip(old_layers, new_layers):
                old.copy_(new)

    def train(self):
        model     = self.model
        optimizer = self.optim
        criterion = self.criter
        epochs    = self.epochs
        traindt   = self.traindt

        if self.gpu:
            model = model.cuda()

        model.train()
        for _ in range(1, epochs+1):
            epoch_loss = 0.0
            for features, target in traindt:
                if self.gpu:
                    features = features.cuda()
                    target   = target.cuda()
                optimizer.zero_grad()

                outputs = model(features)
                loss    = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    def test(self, testloader=None):
        '''Test model using function criterion.'''
        model      = self.model
        criter     = self.criter
        testloader = testloader or self.testdt

        if self.gpu:
            model = model.cuda()

        with torch.no_grad():
            loss = 0.0
            acc  = 0.0
            for features, target in testloader:
                if self.gpu:
                    features = features.cuda()
                    target = target.cuda()

                output = model(features)
                loss  += criter(output, target).item()
                pred   = output.data.max(1, keepdim=True)[1]
                acc  += pred.eq(target.data.view_as(pred)).sum()

        size = len(testloader.dataset)
        return { 'loss': loss/size, 'acc': acc/size}
