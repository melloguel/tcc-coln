# coding: utf-8
'''
Combined Learning Algorithm.

To use this lib, the user must create a concrete implementation of the AbstracModel.
Then, create i models, assinging a relevance r_i, 0 < r_i < 1 with sum(r_i) == 1.
Finally, call combine() passing the models and relevance list.
'''

from math import e, prod
from random import choice

### MAIN FUNCTIONS
def inn_create_model(layers, porcs, conv):
    '''
    Combine weights in each layers to compute new weights with respect to weights porcs.
    weights: list of the weights of each model
    porcs:   list of the distribution or relevance of each model
    conv:    parameter used to combine the models

    Assume the following is valid:
    1) len(weights) == len(porcs)
    2) len(weights[0]) == len(weights[1]) == ... == len(weights[n])
    3) weights[0][i].shape == weights[1][i].shape == ... == weights[n][i].shape
    4) sum(porcs) == 1
    '''
    # Group by layer
    layers = [[layer[i] for layer in layers] for i in range(len(layers[0]))]
    layers_inn = []
    for layer in layers:
        avglayer = inn_combine_layers(layer, porcs, conv)
        layers_inn.append(avglayer)
    return layers_inn

def inn_combine_layers(layers, porcs, conv):
    '''
    Combine layers into a new layer accordingly with each one relevance.
    layers: list of layers
    porcs:  list of the distribution or relevance of each model
    conv:   parameter used to combine the models
    '''
    def weighted_sum(x_values, w_values):
        return sum(map(prod, zip(x_values, w_values)))

    def mult_l2_diff(tensors, sum_result):
        n = len(tensors)
        ret = 0.0
        for i in range(n):
            for j in range(i+1, n):
                ret += (tensors[i] - tensors[j]).pow(2)

        ret = ret.sum() if sum_result else ret
        ret = ret.sqrt()
        return ret

    def mult_euclidean_distance(tensors):
        return mult_l2_diff(tensors, True)

    def mult_abs_diff(tensors):
        return mult_l2_diff(tensors, False)

    threshold = mult_euclidean_distance(layers) / prod(layers[0].shape)
    layers    = [layer * porc for layer, porc in zip(layers, porcs)]
    n4_1      = mult_abs_diff(layers)
    n4_sum    = weighted_sum(layers, [e**(conv*porc) for porc in porcs])

    le_threshold = n4_1 <= threshold
    gt_threshold = n4_1 >  threshold

    newlayer = weighted_sum([le_threshold, gt_threshold], [n4_sum + n4_1, n4_sum])

    return newlayer

def combine(**args):
    '''
    Run CoLn [steps] times.

    models:               list of concrete implementations of AbstractModel
    distribution:         relevance of each model
    conv:                 parameter used to merge the weights of a group of layers
    iterations:           number of combinations to be performed
    before_combine_hook:  function (models) -> () to be called
                           before the models are combined into one (before step 3)
    after_combine_hook:   function (models) -> () to be called
                           after the models have their weights combined (after step 3)

    Steps of CoLn Algorithm:
    1) Train each model on their private dataset
    2) Average the models weights
    3) Assign the averaged weight to the weights of each model
    '''

    models              = args['models']
    distribution        = args['distribution']
    conv                = args['conv']
    steps               = args['steps']
    before_combine_hook = args.get('before_combine_hook', None)
    after_combine_hook  = args.get('after_combine_hook', None)
    
    # Check if parameters are ok
    assert len(models) > 1, "There must be at least 2 models"
    assert len(models) == len(distribution), "Each model must have a related distribution"
    assert abs(1 - sum(distribution)) < 1e-7, "The distributions must sum to 1.0"

    # First, ensure models start at the same position
    layers = choice(models).get_layers()
    for model in models:
        model.set_layers(layers)

    # Second, CoLn loop: train, combine, update
    for _ in range(steps):
        # Train
        layers = []
        for model in models:
            model.train()
            layers.append(model.get_layers())

        if before_combine_hook:
            before_combine_hook(models)

        # Combine
        averaged_layers = inn_create_model(layers, distribution, conv)

        # Update
        for model in models:
            model.set_layers(averaged_layers)

        if after_combine_hook:
            after_combine_hook(models)
