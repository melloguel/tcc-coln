#! /usr/bin/env python

import argparse
import re
from pathlib import Path
from os import path
from statistics import NormalDist

from myutils.generic_utils import pre_process_test_result

def conf_interval(data, confidence=0.95):
    """
    Confidence interval from sample data

    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data    
    """
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return dist.mean, h

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process test data.')
    parser.add_argument('--files', type=str, nargs='+',
                        help='Files to be processed')
    parser.add_argument('--no-preprocess', help='Preprocess the data first',
                        action='store_false')

    args = parser.parse_args()
    if args.files:
        print("test,mean,width")
        for fle in args.files:
            values = pre_process_test_result(Path(fle).absolute(), parse=args.no_preprocess)
            mean, width  = conf_interval(values)
            print(f"{fle},{mean:.3},{width:.3}")
