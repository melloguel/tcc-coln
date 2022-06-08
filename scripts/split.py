#! /usr/bin/env python3
import argparse
import json
from math import floor
import random
import re
from time import time
from pathlib import Path
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split

def union_dfs(dfs):
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_all_data(path="./metadata"):
    """
    Load shattered metadata from path
    """
    p   = Path(path)
    dfs = [ pd.read_csv(m) for m in p.iterdir() if m.is_file() ]
    return union_dfs(dfs)

def split_in(df, samples=2):
    """
    Split df in other samples df
    Preserves data distribution
    """
    indexes = list(df.index)
    l       = len(indexes)
    chunks  = l // samples
    res = []
    at  = []
    random.shuffle(indexes)
    for i, e in enumerate(indexes):
        at.append(e)
        if (i + 1) % chunks == 0:
            res.append(at)
            at = []
    if at: res.append(at)
    vs = res[:samples]
    us = res[samples:]
    return [ df.iloc[idx] for idx in vs ], [ df.iloc[idx] for idx in us ]

def tvt_split(dfs, tp, vp, seed):
    """
        dfs       : dataframes to split
        tp        : train data proportion
        vp        : validation propotion
    """
    dfs = dfs if isinstance(dfs, list) else [dfs]
    for df in dfs:
        train, rem = train_test_split(df,  train_size=tp, random_state=seed)
        valid, test  = train_test_split(rem, test_size=vp, random_state=seed)
        split = { 'train'       : train,
                  'eval'        : valid,
                  'test'        : test
                }
        yield split

def write_metadata(splits, template, cols=None):
    """
        spltis: List[dicts]
        template: directory prefix
        cols:     columns that should be written
    """
    name =  "./" + template
    Path(name).mkdir(parents=True, exist_ok=True)
    metadata_paths = []
    for i, split in enumerate(splits):
        cname  = name + '/part-' + str(i)
        cfname = cname + '/' + 'metadata_'
        Path(cname).mkdir(parents=True, exist_ok=True)
        paths = dict()
        for data in ['train', 'eval', 'test']:
            final_name = cfname + data + '.csv'
            if cols is not None:
                split[data].to_csv(final_name, index=False, columns=cols)
            else:
                split[data].to_csv(final_name, index=False)
            paths[data] = str(Path(final_name).resolve())

        metadata_paths.append((cname, paths))
        print("wrote:", str(Path(cname).resolve()))

    return metadata_paths

def split_by(df, column, value, percent=0.8):
    indexes  = list(df.loc[df[column] == value].index)
    nindex   = len(indexes)
    chunk = floor(0.80 * (df[column] == value).sum())
    print("chunk =", chunk)
    random.shuffle(indexes)
    res = []
    for i in indexes:
        if len(res) < chunk:
            res.append(i)

    # print(df.iloc[res])
    us = set(res)
    vs = set(df.index) - us
    return [df.iloc[list(us)], df.iloc[list(vs)]]

    # random.sample()
    # return "Not implemented"

def split_in2_by(df, column, value):
    indexes = list(df.index)
    hv      = list(df.loc[df[column] == value].index)
    x       = df.iloc[list(hv)].reset_index()
    xs      = df.iloc[list(set(indexes).difference(set(hv)))].reset_index()
    return x, xs

def shuffle_and_mix(x, y, p=0.5):
    assert 0 < p < 1, "0 < p < 1, but got {}".format(p)
    x_id = list(x.index)
    y_id = list(y.index)

    m = floor(p*len(x_id))

    much_x = list(random.sample(x_id, m))
    much_y = list(random.sample(y_id, m))

    xp     = x.iloc[much_x].reset_index()
    xs     = x.iloc[list(set(x_id).difference(set(much_x)))].reset_index()

    yp     = y.iloc[much_y].reset_index()
    ys     = y.iloc[list(set(y_id).difference(set(much_y)))].reset_index()

    return [union_dfs([xp, ys]), union_dfs([yp, xs])]


def create_config(cfg_path, metadata_paths, dataset_path, log_path):
    """
    Create config from template, so it can use the new metada data
    cfg_path : str, where config.json is
    metadata_paths : [dic(str, str)], location of metadata
    dataset_path   : str, where the data is located
    """

    with open(cfg_path, 'r') as f:
        input_str = f.read()

    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    cfg = json.loads(input_str)

    for dname, paths in metadata_paths:

        tcf = cfg['train_config'].copy()
        tcf['logs_path'] = log_path + dname[1:] + f"/{cfg['seed']}/"
        dts = cfg['dataset'].copy()
        for e in ['train', 'eval', 'test']:
            dts[e + '_csv'] = paths[e]
            dts[e + '_data_root_path'] = dataset_path + '/' if dataset_path[-1] != '/' else dataset_path

        new_cfg = cfg.copy()
        new_cfg['dataset'] = dts
        new_cfg['train_config'] = tcf

        output_str = json.dumps(new_cfg, indent=4, sort_keys=True)
        file_name  = str(Path(dname).resolve()) + '.json'
        with open(file_name, 'w') as f:
            f.write(output_str)
        print('wrote:', file_name)

def tests(df):
    dfs, rest = split_in(df, 3)
    dfs1 = split_by(df, 'sexo', 'M', 0.8)
    assert pd.concat(dfs1).sort_index().equals(df)
    assert pd.concat(dfs + rest).sort_index().equals(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metapath", help="metadata path")
    parser.add_argument("column", type=str, help="column to be split")
    parser.add_argument("value", type=int, help="value to be choosen")
    # parser.add_argument("train_size", type=float, help="train data porcentage")
    # parser.add_argument("eval_size", type=float, help="validation data porcentage")
    parser.add_argument("template", help="directories names prefix")
    parser.add_argument("configpath", help="configuration file as template")
    parser.add_argument("logpath", help="log path")
    parser.add_argument("seed", type=int, help="seed to split data")
    # parser.add_argument("dspath", help="dataset path")
    

    args       = parser.parse_args()
    path       = args.metapath
    column     = args.column
    value      = args.value
    train_size = 0.7 # args.train_size
    eval_size  = 0.1 # args.eval_size
    template   = args.template
    cfg_path   = str(Path(args.configpath).resolve())
    ds_path    = str(Path(path).resolve())
    log_path   = str(Path(args.logpath).resolve())
    seed       = args.seed

    print("seed:", seed)
    random.seed(seed)

    # Obtém todo o dataset
    df    = fetch_all_data(path)
    cols  = df.columns
    # Separa em dois datasets, um onde class == 0 e outro onde class != 0
    x, xs = split_in2_by(df, column, value)
    # Passa 'p' dados de um conjunto para o outro
    dfs   = shuffle_and_mix(x, xs, p=0.1)
    # dfs = shuffle_and_mix(x, xs, p=0.2)
    # dfs = shuffle_and_mix(x, xs, p=0.3)
    # dfs = shuffle_and_mix(x, xs, p=0.4)
    # dfs = shuffle_and_mix(x, xs, p=0.5)
    # Separa cada dataset em partes de treino, validação e teste
    splits = list(tvt_split(dfs, train_size, eval_size, seed))
    # Escreve os metadados em disco
    metadata_paths = write_metadata(splits, template, cols)
    # Cria arquivo de configuração para usar os novos metadados
    create_config(cfg_path, metadata_paths, ds_path, log_path)

    # LEGADO
    # tests(df)
    # dfs, rest  = split_in(df, args.n)
    # dfs = split_by(df, 'class', 0, 0.33)
    # dfs = split_by(df, 'class', 0, 1)

