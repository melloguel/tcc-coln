"""
Utilities to read and write configurations in json format.
"""

import json
import re

def load_config(config_path):
    """
    Load config.json as a python dictionary
    """
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    cfg  = json.loads(input_str)
    return cfg


def save_config(cfg, path):
    """
    Write the configuration (cfg) as .json file in path.
    """
    with open(path, 'w') as out:
        json.dump(cfg, out, indent=4, sort_keys=True)

def pre_process_test_result(file_path, parse):
    with open(file_path, 'r') as fle:
        contents = fle.readlines()

    data = map(lambda s: re.search(r"Acurracy:\s+(.*)$", s), contents) if parse else contents
    data = filter(lambda i: i is not None, data)
    data = map(lambda m: float(m.groups()[0]), data) if parse else map(float, data)
    data = list(data)

    return data
