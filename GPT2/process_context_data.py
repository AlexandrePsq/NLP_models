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
from tokenizer import tokenize
from model import GPT2Extractor
from gpt2_utils import set_seed
from dataset import Dataset, InputExample, InputFeatures


#### Functions ####
def f(i, sequence, max_seq_length, set_type):
    guid = "%s-%s" % (set_type, i)
    text_a = pad_to_max_length([0] + sequence + [225, 2], max_seq_length)
    text_b = None
    label = text_a
    example = InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label)
    return example

def pad_to_max_length(sequence, max_seq_length):
    """Pad sequence to reach max_seq_length"""
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    result = sequence + [225, 1] * ((max_seq_length - n)// 2)
    if len(result)==max_seq_length:
        return result
    else:
        return result + [225]

def read_file(filename):
    """Read a pickled object."""
    with open(filename, 'rb') as inp:  
        data = pickle.load(inp)
    return data

def save_file(filename, data):
    """Read a pickled object."""
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

#### Variables ####
data_path = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/all_training' # path to data folder
language = 'english'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parallel processing of training data for controled-contxet analysis.')
    parser.add_argument("--set_type", type=str)
    parser.add_argument("--index", type=int)
    parser.add_argument("--divide_into", type=int)
    parser.add_argument("--split", type=int) 
    parser.add_argument("--context_size", type=int)
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--merge", type=bool, default=False)    
    
    args = parser.parse_args()
    
    max_seq_length = args.max_seq_length if args.max_seq_length is not None else args.context_size + 5
    
    if args.merge:
        print('Merging')
        files = sorted(glob.glob(os.path.join(data_path, f'gpt2_context-{args.context_size}_{args.set_type}_examples_split-{args.split}_*.pkl')))
        print(f'Loading {len(files)} files...')
        data = [read_file(filename) for filename in tqdm(files)]
        print(f'{len(data)} files with {len(data[0])} object in them.')
        print('Flattening...')
        data = [i for l in data for i in l]
        print('Saving...')
        save_file(os.path.join(data_path, f'gpt2_context-{args.context_size}_{args.set_type}_examples_split-{args.split}.pkl'), data)
        print('Merged.')
        print('Cleaning...')
        for filename in files:
            os.remove(filename)
        print('Done.')
        
    else:
    
        filename = os.path.join(data_path, f'gpt2_context-{args.context_size}_{args.set_type}_all-ids_split-{args.split}.pkl')

        data = read_file(filename)

        n = len(data)
        start = args.index * n // args.divide_into
        stop = (args.index + 1) * n // args.divide_into

        data = data[start:stop]
        print(f'Processing {len(data)}...')

        examples = [f(i, data[i:i + args.context_size + 2], args.max_seq_length, args.set_type) for i, _ in tqdm(enumerate(data[:-args.context_size -2]))]
        # +1 because we include the current token 
        # and +1 because we want to predict the following token that has to be included...

        save_file(os.path.join(data_path, f'gpt2_context-{args.context_size}_{args.set_type}_examples_split-{args.split}_{args.index}.pkl'), examples)
        
        print('Done')

