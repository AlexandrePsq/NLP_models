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
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import torch.nn.functional as F

#########################################
########## Training parameters ##########
#########################################

def get_preference_params():
    """Default parameters for LSTM training.
    """
    result = {
        'seed': 1111,
        'eval_batch_size': 128,
        'bsz': 128,
        'bptt': 35, # sequence length,
        'clip': 0.25, # gradient clipping,
        'log_interval': 400, # report interval,
        'lr': 20, # learning rate,
        'epochs': 20,
        'shift_surprisal': 0,
        'cuda': torch.cuda.is_available()
    }    
    return result

#########################################
############ Basic functions ############
#########################################

def write(path, text, end='\n'):
    """Write in the specified text file."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)
        

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



###############################################################################
# RNN Utilities
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def batchify_text_with_memory_size(iterator, memory_size):
    final_iterator = []
    final_iterator += iterator[:memory_size]
    for index, _ in enumerate(iterator[memory_size:]):
        final_iterator += iterator[1 + index:1 + index+memory_size]
    return final_iterator


def save(model, data_name, language, path2derivatives, extra_name=''):
    path = '_'.join([model.__name__(), data_name, language]) + f'{extra_name}.pt'
    path = os.path.join(path2derivatives, 'fMRI/models', language, path)
    with open(path, 'wb') as f:
        torch.save(model, f)


def load(model, data_name, language, path2derivatives, extra_name=''):
    path = '_'.join([model.__name__(), data_name, language]) + f'{extra_name}.pt'
    path = os.path.join(path2derivatives, 'fMRI/models', language, path)
    assert os.path.exists(path)
    with open(path, 'rb') as f:
        return torch.load(f)



###############################################################################
# Extracting advanced features
###############################################################################

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None): 
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, 1) #dim modified from 1 to 2

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cy_tilde = torch.tanh(cy_tilde)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cy_tilde)
    hy = outgate * torch.tanh(cy)

    return {'hidden': hy, 'cell': cy, 'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}


def apply_mask(hidden_l, mask):
    if type(hidden_l) == torch.autograd.Variable:
        return hidden_l * mask
    else:
        return tuple(h * mask for h in hidden_l)


def forward(self, input, hidden, param, mask=None):
    weight = self.all_weights
    dropout = param['dropout']
    # saves the gate values into the rnn object
    last_gates = []

    hidden = list(zip(*hidden))

    for l in range(param['nlayers']):
        hidden_l = hidden[l]
        if mask and l in mask:
            hidden_l = apply_mask(hidden_l, mask[l])
        # we assume there is just one token in the input
        gates = LSTMCell(input[0], hidden_l, *weight[l])
        hy = (gates['hidden'], gates['cell'])
        if mask and l in mask:
            hy = apply_mask(hy, mask[l])

        last_gates.append(gates)
        input = hy[0]

        if dropout != 0 and l < param['nlayers'] - 1:
            input = F.dropout(input, p=dropout, training=False, inplace=False)

    self.gates =  {key: torch.cat([last_gates[i][key].unsqueeze(0) for i in range(param['nlayers'])], 0) for key in ['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']}
    self.hidden = {key: torch.cat([last_gates[i][key].unsqueeze(0) for i in range(param['nlayers'])], 0) for key in ['hidden', 'cell']}
    # we restore the right dimensionality
    input = input.unsqueeze(0)
    return input, (self.hidden['hidden'], self.hidden['cell'])