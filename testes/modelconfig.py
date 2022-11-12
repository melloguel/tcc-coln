# coding: utf-8
'''
Setup to train classifiers.
'''
import torch
from model import AbstractModel

class ModelConfig(AbstractModel):
    def __init__(self, **args):

        self.device      = args.get('device', 'cpu')
        self.model       = args['model'].to(self.device)
        self.criter      = args['criterion']
        self.optimizer   = args['optimizer'](self.model.parameters(), **args['optimizer_params'])
        self.traindt     = args['traindt']
        self.validdt     = args['validdt']
        self.testdt      = args['testdt']
        self.epochs      = args['epochs']
        scheduler_params = args['scheduler_params']
        self.scheduler   = args['scheduler']
        self.scheduler   = self.scheduler(self.optimizer, **scheduler_params)

    def get_layers(self):
        layers = []
        for layer in self.model.parameters():
            layers.append(layer.clone())
        return layers

    def set_layers(self, new_layers):
        old_layers = self.model.parameters()
        with torch.no_grad():
            for old, new in zip(old_layers, new_layers):
                old.copy_(new.to(self.device))

    def epoch(self):
        self.model.train()
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(self.traindt):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criter(output, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss/len(self.traindt.dataset)

    def train(self):
        metrics = []
        print('epoch', 'train-loss', 'valid-loss', 'valid-acc')
        for epoch in range(1, self.epochs + 1):
            train_loss = self.epoch()
            valid_acc, valid_loss = self.test(testloader=self.validdt)
            #self.scheduler.step()
            metric = {
                'epoch' : epoch,
                'train-loss' : train_loss,
                'valid-loss' : valid_loss, 
                'valid-acc': valid_acc
            }
            metrics.append(metric)
            print(f'{metric["epoch"]:2d} {metric["train-loss"]:10.5f} {metric["valid-loss"]:10.5f} {metric["valid-acc"]:10.5f}')
        return metrics

    def test(self, testloader=None):
        '''Test model using function criterion.'''

        self.model.eval()
        testloader = testloader or self.testdt
        loss = 0.0
        acc  = 0.0
        with torch.no_grad():
            for x, y in testloader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    loss  += self.criter(output, y).item()

                    pred   = output.argmax(1, keepdim=True)
                    acc  += pred.eq(y.data.view_as(pred)).sum().item()

        size = len(testloader.dataset)
        return acc/size, loss/size

class ModelConfigEnsemble:
    def __init__(self, model, traindt, validdt, testdt, epochs):
        self.model = model
        self.traindt = traindt
        self.validdt = validdt
        self.testdt  = testdt
        self.epochs  = epochs

    def get_layers(self):
        return []

    def set_layers(self, trash):
        return

    def train(self, debug=False):
        return self.model.fit(self.traindt, epochs=self.epochs)
    
    def test(self, debug=False):
        acc, loss = self.model.evaluate(self.testdt, return_loss=True)
        return acc/100, loss
