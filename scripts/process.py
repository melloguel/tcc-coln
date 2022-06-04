#! /usr/bin/env python3

import argparse
import re
import os.path as path
from statistics import NormalDist

def conf_interval(data, confidence=0.95):
    """
    Confidence interval from sample data

    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data    
    """
    dist = NormalDist.from_samples(data)
    z = NormalDist().inv_cdf((1 + confidence) / 2.)
    h = dist.stdev * z / ((len(data) - 1) ** .5)
    return dist.mean - h, dist.mean + h, dist.mean

def pre_process(file_path, parse):
    with open(file_path, 'r') as f:
        contents = f.readlines()

    data = map(lambda s: re.search("Acurracy:\s+(.*)$", s), contents) if parse else contents
    data = filter(lambda i: i is not None, data)
    data = map(lambda m: float(m.groups()[0]), data) if parse else map(float, data)
    data = list(data)

    return data

def main():

    parser = argparse.ArgumentParser(description='Process test data.')
    parser.add_argument('--files', type=str, nargs='+',
                        help='Files to be processed')
    parser.add_argument('--no-preprocess', help='Preprocess the data first',
                        action='store_false')
    parser.add_argument('-s', help='Omit file names', action='store_true')

    args = parser.parse_args()
    # TODO: Setup CSV format to help formatting using column and tr or grep
    if args.files:
        for f in args.files:
            data = pre_process(path.abspath(f), parse=args.no_preprocess)
            l, r, m  = conf_interval(data)
            p = f"{f + ': ' if not args.s else '':<60}" if not args.s else ""
            print(f"{p}{l:<10.4} {r:<10.4} {m:<10.4}")

main()
