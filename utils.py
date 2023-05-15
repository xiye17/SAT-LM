
import os
import argparse
import sys
import json
import re

import numpy as np

from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join
import itertools

def read_jsonline(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if (not x.startswith("#")) and len(x) > 0]
    return [json.loads(x) for x in lines]

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def flatten_nested_list(x):
    return list(itertools.chain(*x))