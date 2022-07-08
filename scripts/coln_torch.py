#!/usr/bin/env python
# coding: utf-8

import math
from math import e
import numpy as np
import torch
from torch import nn
from torch import optim

### HELPER FUNCTIONS
def weighted_sum(x, wx, y, wy):
    return wx * x + wy * y

def torch_euclidean_distance(tensor_1, tensor_2):
    '''
    Compute euclidean distance between tensor_1 and tensor_2.
    Expect both tensors to have the same shape.
    '''
    dist = (tensor_1 - tensor_2).pow(2).sum().sqrt()
    return dist

def torch_abs_diff(tensor_1, tensor_2):
    '''
    Compute absolute difference elem by elem.
    Expect both tensors to have the same shape.
    '''
    diff    = tensor_1 - tensor_2
    absdiff = torch.abs(diff)
    return absdiff

### MAIN FUNCTIONS
def created_model_inn(weightslist, porcs, euler, conv):
    '''
    Blend layers to compute new weights with respect to weights porcs.
    '''
    layers      = weightslist[0].keys()
    weights_inn = {}

    for layer in layers:
        layerweight_1, layerweight_2 = weightslist[0][layer], weightslist[1][layer]
        weights_inn[layer] = inn_calculate_weights(layerweight_1,
                                                   layerweight_2,
                                                   porcs[0],
                                                   porcs[1],
                                                   euler, conv)
    return weights_inn

def inn_calculate_weights(weight_1, weight_2, p_1, p_2, euler, conv):
    # Compute threshold with eucledian distance
    threshold = torch_euclidean_distance(weight_1, weight_2) / weight_1.shape[0]

    weight_1 = weight_1 * p_1
    weight_2 = weight_2 * p_2

    n4_sum = None
    n4_1   = torch_abs_diff(weight_1, weight_2)

    if euler == 0:
        n4_sum = weighted_sum(weight_1, 1 + p_1,
                              weight_2, 1 + p_2)
    elif euler == 1:
        alpha = 0.5*(1 + math.sqrt(5))
        n4_sum = weighted_sum(weight_1, alpha**p_1,
                              weight_2, alpha**p_2)
    else:
        n4_sum = weighted_sum(weight_1, math.exp(conv*p_1),
                              weight_2, math.exp(conv*p_2))

    # Trying to vectorize operations
    # Let's separate in two cases:
    # First, select the elements that are less or equal than threshold
    le_threshold = n4_1 <= threshold
    # Second, select the elemets that are greater than threshold
    gt_threshold = n4_1  > threshold

    # return the result
    return weighted_sum(le_threshold, n4_sum + n4_1,
                        gt_threshold, n4_sum)

def inn_calculate_bias(a, b, p_1, p_2, conv):
    # TODO: This function looks like it is just the else case of inn_calculate_weights
    inn_calculate_weights(a, b, p_1, p_2, 2, conv)

def sum_scaled_weights(weights, conv=1):
    '''
    Return the sum of the listed scaled weights.
    This is equivalent to scaled avg of the weights.
    '''
    porcs = [1/len(weights) for _ in weights]
    avggrad = created_model_inn(weights, porcs, 2., conv)
    return avggrad

class SimpleMLP(nn.Module):
    def __init__(self):
        # super(SimpleMLP, self).__init__()
        super().__init__()
        self.l1    = nn.Linear(in_features=30, out_features=16)
        self.d1    = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.l2    = nn.Linear(in_features=16, out_features=16)
        self.d2    = nn.RelU()
        self.drop2 = nn.Dropout(0.25)
        self.d3    = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.d1(x)
        x = self.drop1(x)
        x = self.l2(x)
        x = self.d2(x)
        x = self.drop2(x)
        x = self.d3(x)
        return x

    def get_weights(self):
        '''Return a dictionary with current weights.'''
        weights = {}
        for i, layer in enumerate(self.modules()):
            if not isinstance(layer, nn.Sequential):
                if getattr(layer, 'weight', None) is not None:
                    weights[i] = layer.weight

        return weights

    def set_weights(self, weights):
        '''Update current weights with given weights.'''
        current_weights = self.get_weights()

        with torch.no_grad():
            for k, weight in current_weights.items():
                weight.copy_(weights[k])

def train(model, trainloader, debug=False):
    '''
    Train model with data in trainloader.
    If debug is True, then print messages showin training progress.
    '''
    # Training parameters
    epochs       = 100
    comms_round  = 20
    learningrate = 0.010

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learningrate,
                          momentum=0.9,
                          weight_decay=learningrate/comms_round)
    if debug:
        print('Started Training!')

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs
            inputs, labels = data

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Training statistics
            running_loss += loss.item()
            if debug and (i % 2000 == 1999):
                print(f'[{epoch + 1}, {i+1:5d} loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    if debug:
        print('Finished Training!')


def test(model, X_test, Y_test, comm_round=None):
    '''
    TODO: Addapt test to pytorch
    Test model in input X_test comparing with Y_test true values.
    '''
    loss, acc = model.evaluate(X_test, Y_test, verbose = 0, batch_size = 128)
    print(f'comm_round: {comm_round} | global_acc: {acc:.3%} | global_loss: {loss}')
    return acc, loss


def coln(model_type, trainloaders):
    '''
    Train a specific model_type using the CoLn algorithm.
    The number of models is determined by len(trainloaders).
    '''

    comms_round = 10

    # First step: create models
    global_model = model_type()
    local_models = [model_type() for _ in trainloaders]

    # Second step: ensure models start in the same point
    scaled_weights = global_model.get_weights()
    for model in local_models:
        model.set_weights(scaled_weights)

    # Third step: loop: train local models, blend weights and update local weights
    for comm_round in range(1, comms_round+1):
        # Train local models and collect weights
        weights = []
        for model, trainloader in zip(local_models, trainloaders):
            train(model, trainloader)
            weights.append(model.get_weights)

        # Blend weights
        average_weights = sum_scaled_weights(weights, conv=0.001)

        # Update local weights with the average weight
        for model in local_models:
            model.set_weights(average_weights)

        # TODO: update global model

    return local_models
