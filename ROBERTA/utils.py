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

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoTokenizer, RobertaTokenizer



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
            quit()
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
########### Specific functions ##########
#########################################

def save(model, tokenizer, output_dir, index):
    """ Saving best-practices: if you use defaults names for the model, 
    you can reload it using from_pretrained().
    """
    output_dir = os.path.join(output_dir, index)
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)

def batchify_per_sentence(iterator, number_of_sentence, pretrained_roberta, max_length=512):
    """Batchify iterator sentence, to get batches of specified number of sentences.
    Arguments:
        - iterator: sentence iterator
        - number_of_sentence: int
    Returns:
        - batch: sequence iterator
        - indexes: tuple of int
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 2 # for special tokens
    tokenizer = AutoTokenizer.from_pretrained(pretrained_roberta)
    
    batch = []
    indexes = []
    sentence_count = 0
    batch_modifications = 0
    n = len(iterator)
    while sentence_count < n:
        stop = min(sentence_count+number_of_sentence, n)
        token_count = len(tokenizer.tokenize(' '.join(iterator[sentence_count:stop]), add_prefix_space=True))
        while token_count > max_length:
            print('WARNING: decreasing number of sentence in a batch to fit max length of {}'.format(max_length))
            batch_modifications += 1
            stop -= 1
            token_count = len(tokenizer.tokenize(' '.join(iterator[sentence_count:stop]), add_prefix_space=True))
        batch.append(' '.join(iterator[sentence_count:stop]))
        indexes.append((0, len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
        sentence_count = stop
    if batch_modifications > 0:
        print('WARNING: {} reductions were done when constructing batches... You should reduce the number of sentence to include.'.format(batch_modifications))
    return batch, indexes

def batchify_per_sentence_with_context(iterator, number_of_sentence, number_sentence_before, pretrained_roberta, max_length=512):
    """Batchify iterator sentence, to get batches of specified number of sentences.
    Arguments:
        - iterator: sentence iterator
        - number_of_sentence: int
        - number_sentence_before: int
    Returns:
        - batch: sequence iterator
        - indexes: tuple of int
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 2 # for special tokens
    assert number_of_sentence > 0
    tokenizer = AutoTokenizer.from_pretrained(pretrained_roberta)
    
    batch = []
    indexes = []
    sentence_count = 0
    batch_modifications = 0
    n = len(iterator)
    if number_sentence_before > 0:
        start = 0
        stop = min(number_sentence_before, n)
        token_count = len(tokenizer.tokenize(iterator[stop], add_prefix_space=True))
        if token_count > max_length:
            raise ValueError('Cannot fit context with additional sentence. You should reduce context length.')
        batch.append(' '.join(iterator[:stop]))
        indexes.append((0, len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
        sentence_count = stop

    while sentence_count < n:
        start = sentence_count - number_sentence_before
        stop = min(sentence_count + number_of_sentence, n)
        token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=True))
        while token_count > max_length:
            print('WARNING: decreasing number of sentence in a batch to fit max length of {}'.format(max_length))
            batch_modifications += 1
            stop -= 1
            token_count = len(tokenizer.tokenize(' '.join(iterator[sentence_count:stop]), add_prefix_space=True))
            if stop==start+number_sentence_before:
                raise ValueError('Too many context sentence. You reach {} tokens only with context.'.format(token_count))
        batch.append(' '.join(iterator[start:stop]))
        item = ' '.join(iterator[start:start+number_sentence_before])
        if item=='':
            indexes.append((0, len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
        else:
            indexes.append((len(tokenizer.tokenize(item, add_prefix_space=True)), len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
        sentence_count = stop
    if batch_modifications > 0:
        print('WARNING: {} reductions were done when constructing batches... You should reduce the number of sentence to include.'.format(batch_modifications))
    return batch, indexes

def batchify_per_sentence_with_pre_and_post_context(iterator, number_of_sentence, number_sentence_before, number_sentence_after, pretrained_roberta, max_length=512):
    """Batchify iterator sentence, to get batches of specified number of sentences.
    Arguments:
        - iterator: sentence iterator
        - number_of_sentence: int
        - number_sentence_before: int
        - number_sentence_after: int
    Returns:
        - batch: sequence iterator
        - indexes: tuple of int
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 2 # for special tokens
    assert number_of_sentence > 0
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta)
    
    batch = []
    indexes = []
    sentence_count = 0
    batch_modifications = 0
    n = len(iterator)
    if number_sentence_before > 0:
        start = 0
        stop = min(number_sentence_before, n)
        token_count = len(tokenizer.tokenize(iterator[stop], add_prefix_space=True))
        if token_count > max_length:
            raise ValueError('Cannot fit context with additional sentence. You should reduce context length.')
        batch.append(' '.join(iterator[:stop]))
        item = batch[-1]
        if item=='':
            indexes.append((0, 0))
        else:
            indexes.append((0, len(tokenizer.tokenize(item, add_prefix_space=True))))
        sentence_count = stop

    while sentence_count < n:
        start = sentence_count - number_sentence_before
        stop = min(sentence_count + number_of_sentence, n)
        stop_post_context = min(stop + number_sentence_after, n)
        token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop_post_context]), add_prefix_space=True))
        while token_count > max_length:
            print('WARNING: decreasing number of sentence in a batch to fit max length of {}'.format(max_length))
            batch_modifications += 1
            stop -= 1
            stop_post_context = min(stop + number_sentence_after, n)
            token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop_post_context]), add_prefix_space=True))
            if stop==start+number_sentence_before:
                raise ValueError('Too many context sentence. You reach {} tokens only with context.'.format(token_count))
        batch.append(' '.join(iterator[start:stop_post_context]))
        item1 = ' '.join(iterator[start:start+number_sentence_before])
        if item1=='':
            indexes.append((0, len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=True))))
        else:
            indexes.append((len(tokenizer.tokenize(item1, add_prefix_space=True)), len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=True))))
        sentence_count = stop
    if batch_modifications > 0:
        print('WARNING: {} reductions were done when constructing batches... You should reduce the number of sentence to include.'.format(batch_modifications))
    return batch, indexes

def batchify(iterator, context_length, pretrained_roberta, max_length=512):
    """Batchify iterator sentence, to get minimum context length 
    when possible.
    Arguments:
        - iterator: sentence iterator
        - context_length: int
    Returns:
        - batch: sequence iterator
        - indexes: tuple of int
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 2 # for special tokens
    tokenizer = AutoTokenizer.from_pretrained(pretrained_roberta)
    
    batch = []
    indexes = []
    sentence_count = 0
    n = len(iterator)
    
    assert context_length < max_length
    token_count = 0
    while sentence_count < n and token_count < max_length:
        token_count += len(tokenizer.tokenize(iterator[sentence_count], add_prefix_space=True))
        if token_count < max_length:
            sentence_count += 1
    batch.append(' '.join(iterator[:sentence_count]))
    indexes.append((0, len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
    
    while sentence_count < n:
        token_count = 0
        sentence_index = sentence_count - 1
        tmp = sentence_count
        while token_count < context_length:
            token_count += len(tokenizer.tokenize(iterator[sentence_index], add_prefix_space=True))
            sentence_index -= 1
        while sentence_count < n and token_count < max_length:
            token_count += len(tokenizer.tokenize(iterator[sentence_count], add_prefix_space=True))
            if token_count < max_length:
                sentence_count += 1
        batch.append(' '.join(iterator[sentence_index+1:sentence_count]))
        indexes.append((len(tokenizer.tokenize(' '.join(iterator[sentence_index+1:tmp]), add_prefix_space=True)), len(tokenizer.tokenize(batch[-1], add_prefix_space=True))))
    return batch, indexes

#########################################
###### Activations related functions ####
#########################################

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent, connection_character='Ġ'):
    '''Aligns tokenized and untokenized sentence given non-subwords "Ġ" prefixed
    Assuming that each subword token that does start a new word is prefixed
    by "Ġ", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.
    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 0
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index+1  < len(tokenized_sent) and (not tokenized_sent[tokenized_sent_index+1].startswith(connection_character)) and tokenized_sent[tokenized_sent_index+1]!='</s>'):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping
    
def extract_activations_from_token_activations(activation, mapping, indexes):
    """Take the average activations of the tokens related to a given word."""
    new_activations = []
    key_start = None
    key_stop = None
    for key_, value in mapping.items(): 
        if (value[0] - 1) == (indexes[0]): #because we added [CLS] token at the beginning
            key_start = key_
    for key_, value in mapping.items(): 
        if value[-1] == (indexes[1]): #because we added [CLS] token at the beginning
            key_stop = key_
    for word_index in range(key_start, key_stop + 1): # len(mapping.keys()) - 1
        word_activation = []
        word_activation.append([activation[:,index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations

def extract_activations_from_special_tokens(activation, mapping):
    """Returns the activations of the special tokens."""
    cls_activations = [activation[:,mapping[0], :].reshape(1,-1)]
    sep_activations = [activation[:,mapping[len(mapping) - 1], :].reshape(1,-1)]
    return cls_activations, sep_activations

def extract_heads_activations_from_special_tokens(activation, mapping):
    """Extract heads activations of each layer for special tokens.
    activation.shape: [nb_layers, nb_heads, sequence_length, hidden_size/nb_heads]"""
    cls_activations = [activation[:, :, mapping[0], :].reshape(1,-1)]
    sep_activations = [activation[:, :, mapping[len(mapping) - 1], :].reshape(1,-1)]
    return cls_activations, sep_activations

def extract_heads_activations_from_token_activations(activation, mapping, indexes):
    """Extract heads activations of each layer for each token.
    Take the average activations of the tokens related to a given word.
    activation.shape: [nb_layers, nb_heads, sequence_length, hidden_size/nb_heads]"""
    new_activations = []
    key_start = None
    key_stop = None
    for key_, value in mapping.items(): 
        if (value[0] - 1)== (indexes[0]): #because we added [CLS] token at the beginning
            key_start = key_
    for key_, value in mapping.items(): 
        if value[-1] == (indexes[1]): #because we added [CLS] token at the beginning
            key_stop = key_
    for word_index in range(key_start, key_stop + 1): # len(mapping.keys()) - 1
        word_activation = []
        word_activation.append([activation[:, :, index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations