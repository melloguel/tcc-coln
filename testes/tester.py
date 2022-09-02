#! /usr/bin/env python3
# TODO: Implement Neural networks to work with MNIST and CIFAR10
# TODO: Implement something to skip tests if they are complete
# TODO: Add argparse, to select test to be run by command line
import argparse
from math import prod
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
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
                      transform=ToTensor())
    validdt = dataset(root='./data',
                      train=False,
                      download=True,
                      transform=ToTensor())

    total = len(traindt)
    traindt, testdt = random_split(
        traindt, lengths=(total - total//6, total//6))
    return traindt, validdt, testdt


def mkpred(i, s):
    def pred(data):
        return i <= data[1] < i+s
    return pred


def mkdl(dt, bs, train=False):
    return DataLoader(dt, batch_size=bs, shuffle=train)


def mkdls(traindt, validdt, testdt, skip, batch_size):
    traindts = []
    validdts = []
    testdts = []
    r = []
    for i in range(0, 10, skip):
        pred = mkpred(i, skip)
        tr = list(filter(pred, traindt))
        tv = list(filter(pred, validdt))
        te = list(filter(pred, testdt))
        r.append(len(tr)/len(traindt))
        traindts.append(mkdl(tr, bs=batch_size, train=True))
        validdts.append(mkdl(tv, bs=batch_size))
        testdts.append(mkdl(te, bs=batch_size))
    return traindts, validdts, testdts, r


def mkhooks(storage, test_gl):
    def bhook(models):
        storage.append(
            ' '.join(f'{model.test(test_gl)["acc"]:.2f}' for model in models))

    def ahook(models):
        storage[-1] += f' {models[0].test(test_gl)["acc"]:.2f}'
        print(storage[-1])

    return bhook, ahook


def wrstorage(storage, testname):
    txt = '\n'.join(storage)
    with open(testname, 'w') as f:
        f.write(txt)


def sanity_test(mkmodel, dataset, testname, batch_size, gpu):
    traindt, validdt, testdt = fetch_dataset(dataset)
    print('size train:', len(traindt))
    print('size valid:', len(validdt))
    print('size test:', len(testdt))
    model = mkmodel(mkdl(traindt, bs=batch_size, train=True),
                    mkdl(validdt, bs=batch_size),
                    mkdl(testdt, bs=batch_size),
                    gpu)
    print('model:', testname)
    print('parameters:', sum(map(prod, map(lambda x: x.shape, model.get_layers()))))
    model.train(debug=True)
    print(model.test())


def test(model, dataset, testname, batch_size, gpu):
    traindt, validdt, testdt = fetch_dataset(dataset)

    cases = {2: 5, 5: 30, 10: 80}

    for n, steps in cases.items():
        traindts, validdts, testdts, r = mkdls(
            traindt, validdt, testdt, 10 // n)

        models = [model(train, valid, test, gpu) for train, valid,
                  test in zip(traindts, validdts, testdts)]

        storage = [' '.join(f'm{i+1}' for i in range(len(models))) + ' gl']
        bhook, ahook = mkhooks(storage, mkdl(testdt, bs=batch_size))

        combine(models,
                r,
                0.01,
                steps=steps,
                before_combine_hook=bhook,
                after_combine_hook=ahook)

        wrstorage(storage, f'{testname}-{n}.dat')
        print(f'\t{testname}-{n} done!')


def main():
    parser = argparse.ArgumentParser(description='Run CoLn tests.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset used for tests. Can be MNIST or CIFAR10.')
    parser.add_argument('--model', type=str, required=True,
                        help='model used in tests. Can be "conv" or "smlp" or "lmlp".')
    parser.add_argument('--check',
                        action='store_true',
                        help='''Test model with full dataset.\
                        May be used to verfiy the neural network performance.''')
    parser.add_argument('--batchsize', dest='batch_size', default=128, type=int,
                        help="Batch size used during training, validation and testing.")
    parser.add_argument('--gpu', action='store_true', help='Use GPU to train the model.')

    datasets = {'MNIST': MNIST,
                'CIFAR10': CIFAR10}

    mnist_networks = {'lmlp': mk_mnist_lmlp,
                      'smlp': mk_mnist_smlp,
                      'conv': mk_mnist_conv
                      }

    # cifar10_networks = { 'lmlp' : mk_cifar10_lmlp,
    #                      'smlp' : mk_cifar10_smlp,
    #                      'conv' : mk_cifcar10_conv
    #                     }

    networks = {'MNIST': mnist_networks,
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
        print('model name should be "conv" or "smlp" or "lmlp"')
        sys.exit(21)

    print(f'Running: {model_name} in dataset {dataset_name}',
          "[check-mode]" if args.check else "[coln-mode]", "batch size:", args.batch_size)
    test_args = (model, dataset, dataset_name.lower() +
                 '-' + model_name, args.batch_size, args.gpu)
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
