import os
import gc
import glob
import torch
try:
    import pickle5 as pickle
except:
    import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as la

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

import utils
from dataset import Dataset, InputExample, InputFeatures 

#### Functions ####


def read_file(filename):
    """Read a pickled object."""
    with open(filename, 'rb') as inp:  
        data = pickle.load(inp)
    return data

def save_file(filename, data):
    """Read a pickled object."""
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
        


i = 3 #, 4
set_type = 'train'

f = "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/all_training/gpt2_context-0_{set_type}_features_split-{i}.pkl"

f = f.format(set_type=set_type, i=i)
data = read_file(f)
n = len(data)//2
data1 = data[:n]
data2 = data[n:]
save_file(f.split('_split-')[0] + f'_split-{2*i}_cut', data1)
save_file(f.split('_split-')[0] + f'_split-{2*i+1}_cut', data2)
