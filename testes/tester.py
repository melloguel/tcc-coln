#! /usr/bin/env python3

import argparse
import csv
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
from sklearn.model_selection import train_test_split

from wisconsin import WISCONSIN

from mnist_smlp import mk_mnist_smlp
from mnist_lmlp import mk_mnist_lmlp
from mnist_conv import mk_mnist_conv
from mnist_mmlp import mk_mnist_mmlp
from mnist_rnn  import mk_mnist_rnn
from mnist_boost import mk_mnist_boost

from cifar10_smlp import mk_cifar10_smlp
from cifar10_mmlp import mk_cifar10_mmlp
from cifar10_conv import mk_cifar10_conv
from cifar10_rnn  import mk_cifar10_rnn
from cifar10_boost import mk_cifar10_boost

from wisconsin_smlp import mk_wisconsin_smlp
from wisconsin_lmlp import mk_wisconsin_lmlp
from wisconsin_mmlp import mk_wisconsin_mmlp
from wisconsin_rnn  import mk_wisconsin_rnn
from wisconsin_boost import mk_wisconsin_boost

import coln
import mean

def fetch_dataset(dataset, transforms):
    transforms = Compose(transforms)

    traindt = dataset(root='./data',
                      train=True,
                      download=True,
                      transform=transforms)

    testdt = dataset(root='./data',
                      train=False,
                      download=True,
                      transform=transforms)

    traindt, validdt = train_test_split(traindt, test_size=0.1)
    return traindt, validdt, testdt


def mkpred(i, s):
    def pred(data):
        return i <= data[1] < i+s
    return pred

def mkdl(dt, bs, train=False):
    return DataLoader(dt, batch_size=bs, shuffle=train)

def mkdls_chaotic(traindt, validdt, testdt, splits, batchsize):
    nclasses = len(set([y for _, y in traindt]))
    skip = nclasses//splits
    traindts = []
    validdts = []
    testdts  = []
    r = []
    for i in range(0, nclasses, skip):
        pred = mkpred(i, skip)
        tr = list(filter(pred, traindt))
        tv = list(filter(pred, validdt))
        te = list(filter(pred, testdt))
        r.append(len(tr)/len(traindt))

        for i in range(0, nclasses):
            pred = mkpred(i, 1)
            cnt = len(list(filter(pred, tr)))
            print(i, ':', cnt) if cnt > 0 else ...
        print('---'*10)

        traindts.append(mkdl(tr, bs=batchsize, train=True))
        validdts.append(mkdl(tv, bs=batchsize))
        testdts.append(mkdl(te, bs=batchsize))

    return traindts, validdts, testdts, r

def mkdls_uniform(traindt, validt, testdt, splits, batchsize):
    nclasses = len(set([y for _, y in traindt]))
    data_sources = [traindt, validt, testdt]
    data_sources_groups = []
    # Split datasets by class
    for source in data_sources:
        d = { label : [] for label in range(nclasses) }
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


#TODO: Remove
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
            ' '.join(f'{model.test(test_gl)[0]:.4f}' for model in models))

    def ahook(models):
        storage[-1] += f' {models[0].test(test_gl)[0]:.4f}'
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
    print('parameters', sum(map(prod, map(lambda x: x.shape, model.get_layers()))) or '-' )
    metrics = model.train()
    acc, loss = model.test()
    print('loss:', loss, 'acc', acc)

    with open(testname+'-test.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['acc', 'loss'])
        writer.writeheader()
        writer.writerow({'acc': acc, 'loss': loss})

    if metrics:
        with open(testname+'.csv', 'w') as f:
            writer = csv.DictWriter(f, metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)


def test(**kwargs):

    model      = kwargs['model']
    split      = kwargs['split']
    dataset    = kwargs['dataset']
    testname   = kwargs['testname']
    batchsize  = kwargs['batchsize']
    device     = kwargs['device']
    transforms = kwargs['transforms']
    combine    = kwargs['combine']
    cases      = kwargs['cases']

    if model([], [], [], 'cpu').get_layers() == []:
        print("Model doesn't support get_layers()")
        return

    traindt, validdt, testdt = fetch_dataset(dataset, transforms)

    for n, steps in cases.items():
        traindts, validdts, testdts, r = split(traindt, validdt, testdt, n, batchsize)

        models = [model(train, valid, test, device) for train, valid,
                  test in zip(traindts, validdts, testdts)]

        storage = [' '.join(f'm{i+1}' for i in range(len(models))) + ' gl']
        bhook, ahook = mkhooks(storage, mkdl(testdt, bs=batchsize))

        combine(models=models,
                distribution=r,
                conv=0.01,
                steps=steps,
                before_combine_hook=bhook,
                after_combine_hook=ahook)

        wrstorage(storage, f'{testname}-{n}.dat')
        print(f'DONE: {testname}-{n}')

def main():
    datasets = {
        'MNIST'    : MNIST,
        'CIFAR10'  : CIFAR10,
        'WISCONSIN': WISCONSIN
    }

    transforms = {
        'MNIST'   : [ToTensor(), Normalize((0.1307,), (0.3081,))],
        'CIFAR10' : [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
        'WISCONSIN': [ToTensor()]
    }

    mnist_networks = {
        'lmlp'  : mk_mnist_lmlp,
        'smlp'  : mk_mnist_smlp,
        'conv'  : mk_mnist_conv,
        'mmlp'  : mk_mnist_mmlp,
        'rnn'   : mk_mnist_rnn,
        'boost' : mk_mnist_boost,
    }

    cifar10_networks = {
        'smlp'  : mk_cifar10_smlp,
        'mmlp'  : mk_cifar10_mmlp,
        'conv'  : mk_cifar10_conv,
        'rnn'   : mk_cifar10_rnn,
        'boost' : mk_cifar10_boost,
    }

    wisconsin_networks = {
        'smlp'  : mk_wisconsin_smlp,
        'lmlp'  : mk_wisconsin_lmlp,
        'mmlp'  : mk_wisconsin_mmlp,
        'rnn'   : mk_wisconsin_rnn,
        'boost' : mk_wisconsin_boost,
    }

    networks = {
        'MNIST'     : mnist_networks,
        'CIFAR10'   : cifar10_networks,
        'WISCONSIN' : wisconsin_networks,
    }

    splitters = {
        'chaotic' : mkdls_chaotic,
        'uniform' : mkdls_uniform,
    }

    combiners = {
        'coln' : coln.combine,
        'mean' : mean.combine,
    }

    cases = {}
    std = {2: 40, 5: 40, 10: 40}
    for dataset_name in datasets.keys():
        cases[dataset_name] = {}
        cases[dataset_name]['uniform'] = std
        cases[dataset_name]['chaotic'] = std
    cases['WISCONSIN']['chaotic'] = { 2: 40 }

    parser = argparse.ArgumentParser(description='Run CoLn tests.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset used for tests. Can be: '+ ', '.join(datasets.keys()))
    parser.add_argument('--model', type=str, required=True,
                        help='model used in tests.')
    parser.add_argument('--splitter', type=str, required=False, default="chaotic",
                        help='how to split dataset for coln training. Can be: '+ ', '.join(splitters.keys()))
    parser.add_argument('--combine', type=str, required=False, default='coln',
                        help='how to combine the models. Default: coln')
    parser.add_argument('--check', action='store_true',
                        help='Test model with full dataset.')
    parser.add_argument('--batchsize', dest='batchsize', default=64, type=int,
                        help="Batch size used during training, validation and testing.")
    parser.add_argument('--no-gpu', action='store_true', default=False, help='Use CPU to train the model.')

    parser.add_argument('--clients', type=int, default=0, help='When not checking, number of clients')

    args = parser.parse_args()

    try:
        dataset_name = args.dataset.upper()
        dataset = datasets[dataset_name]
    except KeyError:
        print(f'dataset {dataset_name} not found. Avaiable datasets are:', ', '.join(datasets.keys()))
        sys.exit(20)

    try:
        model_name = args.model.lower()
        model = networks[dataset_name][model_name]
    except KeyError:
        print(f'model "{model_name}" not found. Available models for {dataset_name} are:', ', '.join(networks[dataset_name].keys()))
        sys.exit(21)

    try:
        splitter = args.splitter.lower()
        split    = splitters[splitter]
    except KeyError:
        print(f'split "{splitter}" not found. Available splits are:', ', '.join(splits.keys()))
        sys.exit(22)

    try:
        combiner = args.combine.lower()
        combine = combiners[combiner]
    except KeyError:
        print(f'combine "{combiner}" not found. Available combine methods are:', ', '.join(combiners.keys()))
        sys.exit(23)

    if args.clients:
        cases[dataset_name][splitter] = { args.clients : 40 }

    testname = dataset_name.lower()+'-'+model_name.lower()
    if not args.check:
        testname += ('-'+args.splitter.lower()+'-'+args.combine.lower())
    device = "cpu" if args.no_gpu else "cuda"
    test_args = {
        'model'      : model,
        'dataset'    : dataset,
        'testname'   : testname,
        'batchsize'  : args.batchsize,
        'device'     : torch.device(device),
        'transforms' : transforms[dataset_name],
        'split'      : split,
        'combine'    : combine,
        'cases'      : cases[dataset_name][splitter],
    }

    if args.check:
        for item, value in test_args.items():
            if isinstance(value, (str, int)):
                print(f'{item:<11}{test_args[item]:>15}')
        sanity_test(**test_args)
    else:
        items = [('model', model_name),
                 ('dataset', dataset_name),
                 ('device', device),
                 ('batch_size', test_args['batchsize']),
                 ('split', splitter),
                 ('combine', combiner),
                 ('cases', cases[dataset_name][splitter])]
        for name, value in items:
            if isinstance(value, dict):
                print(f"{name:<20}", ' '.join(map(lambda x: f"{x[0]}-{x[1]}", value.items())))
            else:
                print(f"{name:<20} {value:<20}")

        test(**test_args)

if __name__ == '__main__':
    main()
