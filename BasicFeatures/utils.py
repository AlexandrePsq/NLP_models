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
import matplotlib.pyplot as plt
from collections import defaultdict

from tokenizer import sentence_to_words

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


def get_function_words_list(language, 
                            path_to_function_words_list='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/WORDRATE/function_words.txt'):
    function_words_list = open(path_to_function_words_list, 'r').read()
    return function_words_list.split('\n')


def create_onsets_files(path_to_onset_folder, nb_runs, n_frames, frame_rate, slice_period):
    for index in range(1, nb_runs + 1):
        length = int((n_frames/frame_rate) // slice_period)
        offsets = np.cumsum(np.ones(length) * slice_period)
        offsets = np.array([round(x, 3) for x in offsets])
        onsets = np.hstack([np.zeros(1), offsets[:-1]])
        duration = np.zeros(length)
        df = pd.DataFrame({})
        df['onsets'] = onsets
        df['offsets'] = offsets
        df['duration'] = duration
        saving_path = 'rms_{}_run{}.csv'.format(slice_period, index)
        df.to_csv(os.path.join(path_to_onset_folder, saving_path), index=False)


#########################################
########## Regressor functions ##########
#########################################


def wordrate(iterator):
    iterator = sentence_to_words(iterator)
    print(len(np.ones(len(iterator))))
    return pd.DataFrame(np.ones(len(iterator)), columns=['wordrate'])

def content_words(iterator, language, path_to_function_words_list):
    iterator = sentence_to_words(iterator)
    function_words_list = get_function_words_list(language, path_to_function_words_list)
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 0 if iterator[index] in function_words_list else 1
    print(len(result))
    return  pd.DataFrame(result, columns=['content_words'])

def function_words(iterator, language, path_to_function_words_list):
    iterator = sentence_to_words(iterator)
    function_words_list = get_function_words_list(language, path_to_function_words_list)
    result = np.zeros(len(iterator))
    for index in range(len(iterator)):
        result[index] = 1 if iterator[index] in function_words_list else 0
    print(len(result))
    return pd.DataFrame(result, columns=['function_words'])

def log_frequency(iterator, language, path_to_lexique_database='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/lexique_database.tsv'):
    """ Returns the logarithm of the word frequency.
    """
    iterator = sentence_to_words(iterator)
    database = pd.read_csv(path_to_lexique_database, delimiter='\t')
    result = np.zeros(len(iterator))
    words = np.array(database['Word'])
    word_with_issues = {
        've': 'have',
        'hadn': 'had',
        'indulgently': 'indulgent',
        'abashed': 'confused',
        'sputtered': 'rapidly',
        'seabird': 'seagull', 
        'gloomily': 'depressive', 
        'grumpily': 'irritable', 
        'panted': 'gasped', 
        'false': 'wrong', 
        'islet': 'isle', 
        'switchman': 'watchmaker', 
        'weathervane': 'weather', 
        'mustn': 'must' 
    }
    for index, word in enumerate(iterator):
        word = word.lower()
        if word in word_with_issues:
            word = word_with_issues[word]
        try:
            result[index] = database['Lg10WF'][np.argwhere(words==word)[0][0]]
        except:
            result[index] = database['Lg10WF'][np.argwhere(words==word.capitalize())[0][0]]
    print(len(result))
    return pd.DataFrame(result, columns=['log_frequency'])
    
def word_position(iterator):
    """Returns word position.
    """
    result = []
    for sentence in iterator:
        result.extend(np.arange(len(sentence.split())))
    print(len(result))
    return pd.DataFrame(result, columns=['word_position'])

def rms(rms_iterator, 
        slice_period=10e-3, 
        path_to_onset_folder='/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/wave/english/onsets-offsets', 
        nb_runs=9):
    """Args:
        - rms_iterator: tuple 
    Returns root mean squared of the audio signal.
    """
    iterator, frame_rate, n_frames, slice_length = rms_iterator
    create_onsets_files(path_to_onset_folder, nb_runs, n_frames, frame_rate, slice_period)
    result = np.apply_along_axis(lambda y: np.sqrt(np.mean(np.square(y, dtype=np.float64))),1, iterator)
    print(len(result))
    return pd.DataFrame(result, columns=['rms'])
