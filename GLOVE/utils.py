import os
import wget
import time
import yaml
import glob
import torch
import random
import inspect
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from numpy import linalg as la
import matplotlib.pyplot as plt
from collections import defaultdict



#########################################
############ Basic functions ############
#########################################

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass
    
def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
    return parameters

def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def filter_args(func, d):
    """ Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    Returns:
        - args: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys if ((key!='self') and (key in d.keys()))}
    return args

def get_device(device_number=0, local_rank=-1):
    """ Get the device to use for computations.
    """
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        if torch.cuda.is_available():
            print('We will use the GPU:', torch.cuda.get_device_name(device_number))
        else:
            print('No GPU available, using the CPU instead.')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def set_seed(value=1111):
    """ Set all seeds to a given value for reproductibility."""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)



#########################################
########### Special functions ###########
#########################################

#def neighborhood_density(model, iterator, method='mean', threshold=0.7, param=None):
#    columns_activations = ['neighborhood_density']
#    activations = []
#    # computing metric
#    result = np.zeros(len(model))
#    tmp = np.zeros((len(model), len(model)))
#    for i in range(len(model) - 1):
#        for j in range(i + 1, len(model)):
#            tmp[i,j] = cosine_similarity(model.vectors[i], model.vectors[j])
#        vector = tmp[0,1:] if i==0 else (concat(tmp[i,i+1:], tmp[:i,i]))
#        if method == 'mean':
#            result[i] = np.mean(vector)
#        elif method == 'threshold':
#            vector[vector < threshold] = 0
#            result[i] = np.count_nonzero(vector)
#    # generating prediction
#    for item in tqdm(iterator):
#        if item in words2add.keys():
#            for word in words2add[item][0]:
#                activations.append(result[model[word].index])
#            skip = words2add[item][1]
#        elif skip ==0:
#            activations.append(result[model[item].index])
#        else:
#            skip -= 1
#    return pd.DataFrame(np.vstack(activations), columns=columns_activations)


def embeddings(model, iterator, embedding_size):
    columns_activations = ['embedding-{}'.format(i) for i in range(1, 1 + embedding_size)]
    activations = []
    for item in tqdm(iterator):
        if item not in model.keys():
            item = '<unk>'
        activations.append(model[item])
    return pd.DataFrame(np.vstack(activations), columns=columns_activations)

def embeddings_past_context(model, iterator, embedding_size, context_size, decreasing_factor, normalize=False):
    columns_activations = ['embedding-{}'.format(i) for i in range(1, 1 + embedding_size)]
    activations = []
    for index, item in tqdm(enumerate(iterator)):
        activation = np.zeros(embedding_size)
        if item not in model.keys():
            item = '<unk>'
        tmp = model[item]/la.norm(model[item], ord=2) if normalize else model[item]
        activation += tmp
        for i, item_context in enumerate(iterator[max(0, index-context_size+1):index]): # +1 because context_size==1 is the current word
            if item_context not in model.keys():
                item_context = '<unk>'
            tmp = model[item_context]/la.norm(model[item_context], ord=2) if normalize else model[item_context]
            activation += tmp * (decreasing_factor ** (len(iterator[max(0, index-context_size+1):index]) - i))
        activations.append(activation/len(iterator[max(0, index-context_size+1):index+1]))
    return pd.DataFrame(np.vstack(activations), columns=columns_activations)

def embeddings_future_context(model, iterator, embedding_size, context_size, decreasing_factor, normalize=False):
    columns_activations = ['embedding-{}'.format(i) for i in range(1, 1 + embedding_size)]
    activations = []
    for index, item in tqdm(enumerate(iterator)):
        activation = np.zeros(embedding_size)
        if item not in model.keys():
            item = '<unk>'
        tmp = model[item]/la.norm(model[item], ord=2) if normalize else model[item]
        activation += tmp
        for i, item_context in enumerate(iterator[min(index+1, len(iterator)): min(index+1 + context_size, len(iterator))]): # +1 because context_size==1 for future is the current word + the next word
            if item_context not in model.keys():
                item_context = '<unk>'
            tmp = model[item_context]/la.norm(model[item_context], ord=2) if normalize else model[item_context]
            activation += tmp * (decreasing_factor ** (i+1))
        activations.append(activation/len(iterator[min(index, len(iterator)): min(index+1 + context_size, len(iterator))]))
    return pd.DataFrame(np.vstack(activations), columns=columns_activations)