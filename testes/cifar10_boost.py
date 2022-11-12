import torch.nn as nn
from torch import optim
from torchensemble import GradientBoostingClassifier

from cifar10_smlp import SMLP

from modelconfig import ModelConfigEnsemble

def mk_cifar10_boost(traindt, validdt, testdt, device):
    ensemble = GradientBoostingClassifier(
            estimator=SMLP,
            n_estimators=10,
            cuda=device=='cuda',
    )

    criterion = nn.CrossEntropyLoss()
    ensemble.set_criterion(criterion)
    ensemble.set_optimizer( "Adam", lr=0.001, weight_decay=5e-4)

    epochs = 10

    return ModelConfigEnsemble(
           model=ensemble,
           traindt=traindt,
           validdt=validdt,
           testdt=testdt,
           epochs=epochs)
