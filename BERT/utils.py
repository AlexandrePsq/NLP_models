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
from transformers import BertForQuestionAnswering, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertForNextSentencePrediction, BertForSequenceClassification, BertForTokenClassification
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForMaskedLM, WEIGHTS_NAME, CONFIG_NAME



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

def get_device(device_number=0):
    """ Get the device to use for computations.
    """
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(device_number))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
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

def fetch_dataset_from_url(url, local_path):
    """ Download dataset from URL.
    COLA: 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    """
    print('Downloading dataset...')
    if not os.path.exists(local_path):
        try:
            wget.download(url, local_path)
        except Exception:
            raise Exception("Invalid URL: {}".format(url))
    # Unzip the dataset (if we haven't already)
    os.system("unzip {} -d ./datasets".format(local_path))
    os.system("rm {}".format(local_path))
    # return path to the new dataset
    list_of_files = glob.glob('./datasets/*')
    latest_folder = max(list_of_files, key=os.path.getctime)
    return latest_folder

def fetch_data(path, local_path=None, **kwargs):
    """ Fetch data.   :)
    """
    train, test = None, None
    if os.path.exists(path):
        try:
            train_file = sorted(glob.glob(os.path.join(path, '*train*')))[0]
            train = pd.read_csv(train_file, engine='python', **kwargs)
        except:
            raise FileExistsError("Verify that your train file is in {}. Name should included 'train'.\nOr modify your yaml file.".format(path))
        try:
            test_file = sorted(glob.glob(os.path.join(path, '*test*')))[0]
            test = pd.read_csv(test_file, engine='python', **kwargs)
        except:
            print("Verify that your test file is in {}. Name should included 'test'.\nOr modify your yaml file.".format(path))
    else:
        local_path = local_path if local_path else "./data"
        latest_folder = fetch_dataset_from_url(path, local_path)
        print('Lastest downloaded dataset: {}\n'.format(latest_folder))
        print(os.listdir(latest_folder))
        train, test = fetch_data(latest_folder, **kwargs)
    return train, test

def save(model, tokenizer, output_dir):
    """ Saving best-practices: if you use defaults names for the model, 
    you can reload it using from_pretrained().
    """
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


#########################################
###### Activations related functions ####
#########################################

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed
    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
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
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[tokenized_sent_index + 1].startswith('##')):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping

def extract_hidden_state_activations_from_tokenized(activation, mapping):
    """Take the average activations of the tokens related to a given word."""
    new_activations = []
    for word_index in range(1, len(mapping.keys())):
        word_activation = []
        word_activation.append([activation[:,index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations

def extract_attention_head_activations_from_tokenized(activation, mapping):
    """"""
    new_activations = []
    for word_index in range(1, len(mapping.keys())):
        word_activation = []
        word_activation.append([activation[:, :, index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations