from math import prod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

#TODO: Balance dataset

class DT(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        try:
            return self.x[idx], self.y[idx].item()
        except:
            return self.x[idx], self.y[idx]

def generate_splits(root, t):
    path = root + '/wisconsin.csv'
    testpath = root + '/wisconsin-test.csv'
    trainpath = root + '/wisconsin-train.csv'

    # Write test and train datasets
    if not (Path(testpath).exists() or Path(trainpath).exists()):
        df = pd.read_csv(path, index_col=[0])
        df = df.dropna(how='all', axis='columns')
        df = df.replace({'diagnosis': { 'M' : 1., 'B': 0. }})

        train, test = train_test_split(df, test_size=0.2) 
        train.to_csv(trainpath, index=False)
        test.to_csv(testpath, index=False)
    
    return pd.read_csv(trainpath if t else testpath)

def WISCONSIN(root, **args):
    df = generate_splits(root, args['train'])
    scaler = StandardScaler()
    x = scaler.fit_transform(df.drop(labels='diagnosis', axis=1))
    y = df['diagnosis'].to_numpy() 

    oversample = SMOTE()
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)
    x, y = oversample.fit_resample(x, y)
    dt = DT(x, y)
    return dt
