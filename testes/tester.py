#! /usr/bin/env python3
#TODO: Implement Neural networks to work with MNIST and CIFAR10
#TODO: Implement something to skip tests if they are complete
#TODO: Add argparse, to select test to be run by command line
import argparse
from math import prod
import sys

import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from mnist_smlp import mk_mnist_smlp
from mnist_lmlp import mk_mnist_lmlp
from mnist_conv import mk_mnist_conv


from coln import combine


def fetch_dataset(dataset):
    traindt = dataset(root='./data',
                      train=True,
                      download=True,
                      transform = ToTensor())
    testdt   = dataset(root='./data',
                       train=False,
                       download=True,
                       transform = ToTensor())
    return traindt, testdt

def mkpred(i, s):
    def pred(data):
        return i <= data[1] < i+s
    return pred

def mkdl(dt):
    return DataLoader(dt, batch_size=128)

def mkdls(traindt, testdt, skip):
    traindts = []
    testdts  = []
    r = []
    for i in range(0, 10, skip):
        pred = mkpred(i, skip)
        tr = list(filter(pred, traindt))
        te = list(filter(pred, testdt))
        r.append(len(tr)/len(traindt))
        traindts.append(mkdl(tr))
        testdts.append(mkdl(te))
    return traindts, testdts, r

def mkhooks(storage, test_gl):
    def bhook(models):
        storage.append(' '.join(f'{model.test(test_gl)["acc"]:.2f}' for model in models))

    def ahook(models):
        storage[-1] += f' {models[0].test(test_gl)["acc"]:.2f}'
        print(storage[-1])

    return bhook, ahook

def wrstorage(storage, testname):
    txt = '\n'.join(storage)
    with open(testname, 'w') as f:
        f.write(txt)

def sanity_test(mkmodel, dataset, testname):
    traindt, testdt = fetch_dataset(dataset)
    model = mkmodel(mkdl(traindt), mkdl(testdt))
    print('test:', testname)
    print('parameters:', sum(map(prod, map(lambda x: x.shape, model.get_layers()))))
    model.train()
    print(model.test())

def test(model, dataset, testname):
    traindt, testdt = fetch_dataset(dataset)

    cases = [2, 5, 10]
    steps = { 2 : 5, 5 : 30, 10 : 80 }

    for n in cases:
        traindts, testdts, r  = mkdls(traindt, testdt, 10 // n)

        models = [model(train, test) for train, test in zip(traindts, testdts)]

        storage = [' '.join(f'm{i+1}' for i in range(len(models))) + ' gl']
        bhook, ahook = mkhooks(storage, mkdl(testdt))

        combine(models,
                r,
                0.01,
                steps=steps[n],
                before_combine_hook=bhook,
                after_combine_hook=ahook)

        wrstorage(storage, f'{testname}-{n}.dat')
        print('\t'+testname, 'done!')

def main():
    parser = argparse.ArgumentParser(description='Run CoLn tests.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset used for tests. Can be MNIST or CIFAR10')
    parser.add_argument('--model', type=str, required=True,
                        help='model used in tests. Can be one of: conv or smlp or lmlp')
    parser.add_argument('--check',
                        action='store_true',
                        help='''Test model with full dataset.\
                        May be used to verfiy the neural network performance.''')

    datasets = { 'MNIST' : MNIST,
                 'CIFAR10' : CIFAR10 }
    
    mnist_networks = { 'lmlp' : mk_mnist_lmlp,
                       'smlp' : mk_mnist_smlp,
                       'conv' : mk_mnist_conv
                      }

    # cifar10_networks = { 'lmlp' : mk_cifar10_lmlp,
    #                      'smlp' : mk_cifar10_smlp,
    #                      'conv' : mk_cifcar10_conv
    #                     }

    networks = { 'MNIST' : mnist_networks,
                 # 'CIFAR10' : cifar10_networks
                 }

    args = parser.parse_args()

    try:
        dataset_name = args.dataset.upper()
        dataset = datasets[dataset_name]
    except KeyError:
        print('dataset must be CIFAR10 or MNIST, got', dataset_name)
        sys.exit(20)

    try:
        model_name = args.model.lower()
        model = networks[dataset_name][model_name]
    except KeyError:
        print('model name should be one of: conv, smlp, lmlp')
        sys.exit(21)

    print(f'Running: {model_name} in dataset {dataset_name}', "[check-mode]" if args.check else "[coln-mode]")
    test_args = (model, dataset, dataset_name.lower() + '-' + model_name)
    if args.check:
        sanity_test(*test_args)
    else:
        test(*test_args)

    # test(mk_mnist_smlp, MNIST, 'mnist-smlp')
    # test(mk_mnist_conv, MNIST, 'mnist-conv')
    # sanity_test(mk_mnist_smlp, MNIST, 'smlp MNIST')
    # sanity_test(mk_mnist_lmlp, MNIST, 'lmlp MNIST')
    # test(mk_mnist_lmlp, MNIST, 'mnist-lmlp')

    # test(mk_cifar10_conv, CIFAR10, 'cifar10-conv')
    # test(mk_cifar10_smlp, CIFAR10, 'cifar10-smlp')
    # test(mk_cifar10_lmlp, CIFAR10, 'cifar10-lmlp')

    # print('Done!')

if __name__ == '__main__':
    main()
