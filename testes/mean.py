# coding: utf-8
from random import choice

### MAIN FUNCTIONS
def mean(layers, distribution):
    layers = [[layer[i] for layer in layers] for i in range(len(layers[0]))]
    layers_inn = []
    for layer in layers:
        avglayer = sum(l*w for l, w in zip(layer, distribution))
        layers_inn.append(avglayer)
    return layers_inn

def combine(**args):

    models              = args['models']
    distribution        = args['distribution']
    conv                = args['conv']
    steps               = args['steps']
    before_combine_hook = args.get('before_combine_hook', None)
    after_combine_hook  = args.get('after_combine_hook', None)

    layers = choice(models).get_layers()
    for model in models:
        model.set_layers(layers)

    for _ in range(steps):
        # Train
        layers = []
        for model in models:
            model.train()
            layers.append(model.get_layers())

        if before_combine_hook:
            before_combine_hook(models)

        # Combine
        averaged_layers = mean(layers, distribution)

        # Update
        for model in models:
            model.set_layers(averaged_layers)

        if after_combine_hook:
            after_combine_hook(models)
