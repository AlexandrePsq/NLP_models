""" Code to fine-tune hugging-face implementation of GPT2 model.
https://huggingface.co/
"""
import warnings
warnings.simplefilter(action='ignore')

import os
import gc
import wget
import time
import yaml
import glob
import torch
import random
import inspect
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import AdamW, GPT2Config, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, WEIGHTS_NAME, CONFIG_NAME, GPT2Config

from language_modeling import LMDataset, LMProcessor
from gpt2_utils import read_yaml, set_seed, format_time, filter_args, get_device, save, check_folder, save_yaml
from processors import DataProcessor, ModelProcessor
from reporting import Report
from dataset import Dataset



########################################################################################################
# ------------------------------------------- FINE - TUNING -------------------------------------------#
########################################################################################################

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Fine-tune a GPT2 model for a specific NLP task.")
    parser.add_argument('--yaml_file', type=str, help='''Path to the yaml file containing additional information on how 
                                                        the dataset is structured.''')
    args = parser.parse_args()

    # Fetch parameters
    parameters = read_yaml(args.yaml_file)
    check_folder(parameters['output_dir'])
    nb_splits = parameters['nb_splits']
    save_yaml(parameters, os.path.join(parameters['output_dir'], 'lpp_config_evaluation.yml'))
    logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)
    logging.info("Parameters fetched.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters['task'].lower()
    logging.info("\tDone.")

    logging.info("Instanciating dataset and data processor...")
    data = LMDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'])
    processor = LMProcessor(parameters['max_length'], device=device, output_dir=parameters['output_dir'], dataset_name=parameters['dataset_name'], dataset_dir=parameters['dataset_dir'], context_size=parameters['context_size'], extra=parameters['extra'], n_splits=nb_splits)
    logging.info("\tDone.")

    paths = sorted(glob.glob(os.path.join(parameters['output_dir'], 'end-epoch*')))
    paths = [p for p in paths if 'split' not in p]
        
    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info("\tDone.")
    
    nb_splits = parameters['nb_splits']
    data_paths = processor.get_data(data, 'dev')

    logging.info("\tDone.")
    
    logging.info("Fine-tuning the model.")
    gc.collect()
    model_processor = ModelProcessor(None, None, None, 
                                        None, device, 
                                        parameters['metric_name'], 
                                        None,
                                        parameters['use_output_mask'],
                                        context_size=parameters['context_size']
                                    )
    testing_stats  = []
    
    try:
        for path in paths:
            print(f'Using model saved at: {path}...')
            logging.info(f'Using model saved at: {path}...')
            logging.info("Setting seed for reproductibility...") 
            set_seed(parameters['seed'])
            logging.info("\tDone.")
            model_processor.model = GPT2LMHeadModel.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
            model_processor.model.to(device)
            accuracy, loss = None, None
            
            logging.info("Evaluation report: ")
            test_loss, test_time = model_processor.evaluate(processor, data_paths, 'dev', parameters) 
            testing_stats.append({
                        'Valid. Loss': test_loss,
                        #'Active Test. Loss': active_test_loss,
                        'Valid. Time': test_time,
                    })
            df = pd.DataFrame(data=testing_stats)
            df.to_csv(os.path.join(parameters['output_dir'], 'validation_stats.csv'), index=False)
            logging.info("\tDone.")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
