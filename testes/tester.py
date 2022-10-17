#! /usr/bin/env python3

import argparse
from math import prod
from itertools import groupby
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader

from mnist_smlp import mk_mnist_smlp
from mnist_lmlp import mk_mnist_lmlp
from mnist_conv import mk_mnist_conv
from mnist_mmlp import mk_mnist_mmlp

from cifar10_smlp import mk_cifar10_smlp
from cifar10_conv import mk_cifar10_conv

from coln import combine

def fetch_dataset(dataset, transforms):
    transforms = Compose(transforms)

    traindt = dataset(root='./data',
                      train=True,
                      download=True,
                      transform=transforms)

    validdt = dataset(root='./data',
                      train=False,
                      download=True,
                      transform=transforms)

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

def mkdls_chaotic(traindt, validdt, testdt, splits, batchsize):
    skip = 10//splits
    traindts = []
    validdts = []
    testdts  = []
    r = []
    for i in range(0, 10, skip):
        pred = mkpred(i, skip)
        tr = list(filter(pred, traindt))
        tv = list(filter(pred, validdt))
        te = list(filter(pred, testdt))
        r.append(len(tr)/len(traindt))
        traindts.append(mkdl(tr, bs=batchsize, train=True))
        validdts.append(mkdl(tv, bs=batchsize))
        testdts.append(mkdl(te, bs=batchsize))
    return traindts, validdts, testdts, r

def mkdls_uniform(traindt, validt, testdt, splits, batchsize):
    data_sources = [traindt, validt, testdt]
    data_sources_groups = []
    # Split datasets by class
    for source in data_sources:
        d = { label : [] for label in range(10) }
        for data in source:
            feature, label = data
            d[label].append(data)
        data_sources_groups.append(d)

    # Split each class in #splits for each dataset
    out = []
    for src_group in data_sources_groups:
        d = {}
        for label, item in src_group.items():
            data = list(item)
            chunk_size = len(data)//splits
            d[label] = []
            for i in range(0, len(data), chunk_size):
                d[label].append(data[i: i + chunk_size])
        out.append(d)

    # For each dataset split in #splits, create #splits datasets
    ret = []
    for src in out:
        final = [[] for _ in range(splits)]
        for k, v in src.items():
            for i in range(splits):
                final[i].append(v[i])
        for i in range(splits):
            final[i] = [item for sublist in final[i] for item in sublist]
        ret.append(final)

    r = [1/len(ret[0]) for _ in ret[0]]
    ret[0] = map(lambda x: mkdl(x, bs=batchsize, train=True), ret[0])
    ret[1] = map(lambda x: mkdl(x, bs=batchsize), ret[1])
    ret[2] = map(lambda x: mkdl(x, bs=batchsize), ret[2])

    return *ret, r


def sanity_check_uniformity():
    tr, vl, ts = fetch_dataset(MNIST, [ToTensor()])
    for i in [2, 5, 10]:
        trs, vls, tss, r = mkdls_uniform(tr, vl, ts, i, 0)
        print(r)
        print('i ==', i)
        print('len trs = ', len(trs))
        for x in trs:
            print('\t', len(x), type(x))

def mkhooks(storage, test_gl):
    def bhook(models):
        storage.append(
            ' '.join(f'{model.test(test_gl)["acc"]:.4f}' for model in models))

    def ahook(models):
        storage[-1] += f' {models[0].test(test_gl)["acc"]:.4f}'
        print(storage[-1])

    return bhook, ahook

def wrstorage(storage, testname):
    txt = '\n'.join(storage) + '\n'
    with open(testname, 'w') as f:
        f.write(txt)

def sanity_test(**kwargs):

    mkmodel    = kwargs['model']
    dataset    = kwargs['dataset']
    testname   = kwargs['testname']
    batchsize  = kwargs['batchsize']
    device     = kwargs['device']
    transforms = kwargs['transforms']

    traindt, validdt, testdt = fetch_dataset(dataset, transforms)
    print('size train', len(traindt))
    print('size valid', len(validdt))
    print('size test ', len(testdt))
    model = mkmodel(mkdl(traindt, bs=batchsize, train=True),
                    mkdl(validdt, bs=batchsize),
                    mkdl(testdt, bs=batchsize),
                    device)
    print('parameters', sum(map(prod, map(lambda x: x.shape, model.get_layers()))))
    model.train(debug=testname+'.dat')
    print('\n'.join(map(lambda x: f'{x[0]:<10}{x[1]:<10}', model.test().items())))

def test(**kwargs):

    model      = kwargs['model']
    splitter   = kwargs['splitter']
    dataset    = kwargs['dataset']
    testname   = kwargs['testname']
    batchsize  = kwargs['batchsize']
    device     = kwargs['device']
    transforms = kwargs['transforms']

    traindt, validdt, testdt = fetch_dataset(dataset, transforms)

    cases = {2: 5, 5: 30, 10: 60}

    for n, steps in cases.items():
        traindts, validdts, testdts, r = splitter(traindt, validdt, testdt, n, batchsize)

        models = [model(train, valid, test, device) for train, valid,
                  test in zip(traindts, validdts, testdts)]

        storage = [' '.join(f'm{i+1}' for i in range(len(models))) + ' gl']
        bhook, ahook = mkhooks(storage, mkdl(testdt, bs=batchsize))

        combine(models,
                r,
                0.01,
                steps=steps,
                before_combine_hook=bhook,
                after_combine_hook=ahook)

        wrstorage(storage, f'{testname}-{n}.dat')
        print(f'DONE: {testname}-{n}')

def main():
    parser = argparse.ArgumentParser(description='Run CoLn tests.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset used for tests. Can be MNIST or CIFAR10.')
    parser.add_argument('--model', type=str, required=True,
                        help='model used in tests.')
    parser.add_argument('--splitter', type=str, required=False, default="chaotic",
                        help='how to split dataset for coln training. Can be "uniform" or "chaotic"')
    parser.add_argument('--check',
                        action='store_true',
                        help='''Test model with full dataset.\
                        May be used to verfiy the neural network performance.''')
    parser.add_argument('--batchsize', dest='batchsize', default=128, type=int,
                        help="Batch size used during training, validation and testing.")
    parser.add_argument('--no-gpu', action='store_false', help='Use CPU to train the model.')

    datasets = {
            'MNIST'  : MNIST,
            'CIFAR10': CIFAR10
    }

    transforms = {
            'MNIST'   : [ToTensor()],
            'CIFAR10' : [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    }

    mnist_networks = {
            'lmlp': mk_mnist_lmlp,
            'smlp': mk_mnist_smlp,
            'conv': mk_mnist_conv,
            'mmlp': mk_mnist_mmlp,
    }

    cifar10_networks = {
            'smlp' : mk_cifar10_smlp,
            #'lmlp' : mk_cifar10_lmlp,
            'conv' : mk_cifar10_conv
    }

    networks = {
            'MNIST': mnist_networks,
            'CIFAR10' : cifar10_networks
    }

    splitters = {
            'chaotic' : mkdls_chaotic,
            'uniform' : mkdls_uniform,
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
        print(f'model "{model_name}" not found. Available models for {dataset_name} are:', ', '.join(networks[dataset_name].keys()))
        sys.exit(21)

    try:
        splitter = splitters[args.splitter.lower()]
    except KeyError:
        print(f'split "{split}" not found. Available splits are:', ', '.join(splits.keys()))

    test_args = {
            'model'      : model,
            'dataset'    : dataset,
            'testname'   : dataset_name.lower()+'-'+model_name.lower()+('-'+args.splitter.lower() if not args.check else ''),
            'batchsize'  : args.batchsize,
            'device'     : "cuda" if not args.no_gpu else "cpu",
            'transforms' : transforms[dataset_name],
            'splitter'   : splitter,
    }

    if args.check:
        for item, value in test_args.items():
            if isinstance(value, (str, int)):
                print(f'{item:<11}{test_args[item]:>15}')
        sanity_test(**test_args)
    else:
        print('testing coln')
        test(**test_args)

if __name__ == '__main__':
    main()
