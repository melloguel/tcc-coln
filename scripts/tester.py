#TODO: Implement Neural networks to work with MNIST and CIFAR10
import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
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

    return bhook, ahook

def wrstorage(storage, testname):
    with open(f'{testname}-{n}.dat', 'w') as f:
        for line in storage:
            f.write(line + '\n')

def test(model, dataset, testname):
    traindt, testdt = fetch_dataset(dataset)

    cases = [2, 5, 10]
    steps = { 2 : 5, 5 : 30, 10 : 60 }

    for n in cases:
        traindts, testdts, r  = mkdls(traindt, testdt, 10 // n)

        models = [model(train, test) for train, test in zip(traindts, testdts)]

        storage = [' '.join(f'm{i+1}' for i in range(len(models)) + ' gl')]
        bhook, ahook = mkhooks(storage, mkdl(testdt))

        combine(models,
                r,
                0.01,
                steps=steps[n],
                before_combine_hook=bhook,
                after_combine_hook=ahook)

        wrstorage(storage, testname)

if __name__ == '__main__':
    print('testing!')
    print('mnist')
    test(mk_mnist_conv, MNIST, 'mnist-conv')
    test(mk_mnist_smlp, MNIST, 'mnist-smlp')
    test(mk_mnist_lmlp, MNIST, 'mnist-lmlp')

    print('cifar10')
    test(mk_cifar10_conv, CIFAR10, 'cifar10-conv')
    test(mk_cifar10_smlp, CIFAR10, 'cifar10-smlp')
    test(mk_cifar10_lmlp, CIFAR10, 'cifar10-lmlp')

    print('Done!')
