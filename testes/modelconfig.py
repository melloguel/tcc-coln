# coding: utf-8
'''
Setup to train classifiers.
'''
import torch
from coln import AbstractModel

class ModelConfig(AbstractModel):
    def __init__(self, model, criterion, optimizer, optimizer_params,
                 traindt, validdt, testdt, epochs, gpu=None):
        self.model   = model
        self.criter  = criterion
        self.optim   = optimizer(model.parameters(), **optimizer_params)
        self.traindt = traindt
        self.validdt = validdt
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

    def one_epoch(self, debug=False):
        running_loss = 0.
        last_loss    = 0.

        mark = len(self.traindt)

        for i, data in enumerate(self.traindt):
            inputs, labels = data
            if self.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.optim.zero_grad()

            outputs = self.model(inputs)

            loss = self.criter(outputs, labels)
            loss.backward()

            self.optim.step()

            running_loss += loss.item()
            if i % (mark//4) == (mark//4 - 1):
                last_loss = running_loss / 1000
                if debug:
                    print(f'    batch {i+1:3d} loss: {last_loss}')
                running_loss = 0.

        return last_loss

    def train(self, debug=False):

        if self.gpu:
            self.model = self.model.cuda()

        self.model.train()

        for epoch in range(self.epochs):
            if debug:
                print(f'EPOCH {epoch+1}')

            self.model.train(True)
            avg_loss = self.one_epoch(debug)

            self.model.train(False)
            running_vloss = 0.0
            for vinputs, vlabels in self.validdt:
                if self.gpu:
                    vinputs = vinputs.cuda()
                    vlabels = vlabels.cuda()

                voutputs = self.model(vinputs)
                vloss = self.criter(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (len(self.validdt))
            if debug:
                print(f'LOSS train {avg_loss} valid {avg_vloss}')

        if self.gpu:
            self.model = self.model.to('cpu')

    def test(self, testloader=None):
        '''Test model using function criterion.'''

        self.model.train(False)
        testloader = testloader or self.testdt
        loss = 0.0
        acc  = 0.0
        for features, target in testloader:
                output = self.model(features)
                loss  += self.criter(output, target).item()

                pred   = output.data.max(1, keepdim=True)[1]
                acc  += pred.eq(target.data.view_as(pred)).sum()

        size = len(testloader.dataset)
        return { 'loss': loss/size, 'acc': acc/size}