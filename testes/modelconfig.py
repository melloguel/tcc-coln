# coding: utf-8
'''
Setup to train classifiers.
'''
import torch
from coln import AbstractModel

class ModelConfig(AbstractModel):
    #def __init__(self, model, criterion, optimizer, optimizer_params, scheduler, scheduler_params,
    #             traindt, validdt, testdt, epochs, device):
    def __init__(self, **args):

        self.device      = args.get('device', 'cpu')
        self.model       = args['model'].to(self.device)
        self.criter      = args['criterion']
        self.optim       = args['optimizer'](self.model.parameters(), **args['optimizer_params'])
        self.traindt     = args['traindt']
        self.validdt     = args['validdt']
        self.testdt      = args['testdt']
        self.epochs      = args['epochs']
        scheduler_params = args.get('scheduler_params', None)
        self.scheduler   = args.get('scheduler', None)
        self.scheduler   = self.scheduler(self.optim, **scheduler_params) if self.scheduler else None

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
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

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

        loss = [('train', 'validation')]


        self.model.train()

        for epoch in range(self.epochs):
            if debug:
                print(f'EPOCH {epoch+1}')

            self.model.train(True)
            avg_loss = self.one_epoch(debug)

            if self.scheduler:
                self.scheduler.step()

            self.model.train(False)
            running_vloss = 0.0
            for vinputs, vlabels in self.validdt:
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = self.model(vinputs)
                vloss = self.criter(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (len(self.validdt))

            if debug:
                print(f'LOSS train {avg_loss} valid {avg_vloss}')
                loss.append((avg_loss, avg_vloss))

        if isinstance(debug, str):
            with open(debug, 'w') as f:
                f.write(' '.join(loss[0]) + '\n')
                f.write('\n'.join(map(lambda x: f'{x[0]:4.3f} {x[1]:4.3f}', loss[1:])))
                f.write('\n')

    def test(self, testloader=None):
        '''Test model using function criterion.'''

        self.model.train(False)
        testloader = testloader or self.testdt
        loss = 0.0
        acc  = 0.0
        for features, target in testloader:
                features, target = features.to(self.device), target.to(self.device)
                output = self.model(features)
                loss  += self.criter(output, target).item()

                pred   = output.data.max(1, keepdim=True)[1]
                acc  += pred.eq(target.data.view_as(pred)).sum()

        size = len(testloader.dataset)
        return { 'loss': loss/size, 'acc': acc/size}
