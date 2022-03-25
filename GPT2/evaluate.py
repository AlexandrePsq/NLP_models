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
    save_yaml(parameters, os.path.join(parameters['output_dir'], 'config_evaluation.yml'))
    logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)
    logging.info("Parameters fetched.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters['task'].lower()
    logging.info("\tDone.")

    logging.info("Instanciating dataset and data processor...")
    data = LMDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'])
    processor = LMProcessor(parameters['max_length'], device=device, output_dir=parameters['output_dir'], dataset_name=parameters['dataset_name'], dataset_dir=parameters['dataset_dir'])
    logging.info("\tDone.")

    logging.info("Fetching data (training + validation) and parameters...")
    data._fetch_dataset()
    data.process_dataset(parameters['todo'])
    logging.info("\tDone.")
            
    tokenizer = ByteLevelBPETokenizer( 
                    lowercase=parameters['lowercase'])
    files = [os.path.join(parameters['dataset_dir'], item) for item in ['gpt2_train.txt', 'gpt2_test.txt', 'gpt2_dev.txt']]
    tokenizer.train( 
                    files, 
                    vocab_size=parameters['vocab_size'], 
                    min_frequency=parameters['min_frequency'], 
                    show_progress=True, 
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.enable_truncation(max_length=512)
    print(tokenizer.encode("<s> The dog ran <mask> outside . <unk> </s> <pad>").tokens) # --> ['<s>', 'Ġ', '<mask>', 'Ġ.', 'Ġ', '<unk>', 'Ġ', '</s>', 'Ġ', '<pad>']
    print(tokenizer.encode("<s> <mask> . <unk> </s> <pad>").ids) # --> [0, 225, 4, 272, 225, 3, 225, 2, 225, 1]

    processor.set_tokenizer(tokenizer)
    #paths = sorted(glob.glob(os.path.join(parameters['output_dir'], 'checkpoint_*')))
    paths = [parameters['model_path']] if parameters['model_path'] is not None else sorted(glob.glob(os.path.join(parameters['output_dir'], 'end-epoch*')))
        
    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info("\tDone.")
    
    nb_splits = parameters['nb_splits']
    examples_paths = processor.get_test_examples(data) if parameters['todo']=='test' else processor.get_dev_examples(data)
    features_paths = processor.convert_examples_to_features(examples_paths, parameters['max_length'], tokenizer, set_type=parameters['todo'])

    logging.info("\tDone.")
    
    logging.info("Fine-tuning the model.")
    gc.collect()
    model_processor = ModelProcessor(None, None, None, 
                                        None, device, 
                                        parameters['metric_name'], 
                                        None,
                                        parameters['use_output_mask'])
    evaluation_stats  = []
    
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
            accuracy, loss, time, report = model_processor.evaluate(processor, features_paths, parameters['todo'], parameters) 
            evaluation_stats.append({
                        'Loss': loss,
                        'Accur.': accuracy,
                        'Time': time,
                        'model_path': path,
                        'report': report})
            df = pd.DataFrame(data=evaluation_stats)
            df.to_csv(os.path.join(parameters['output_dir'], f"{parameters['output_name']}_evaluation.csv"), index=False)
        df = pd.DataFrame(data=evaluation_stats)
        df.to_csv(os.path.join(parameters['output_dir'], f"{parameters['output_name']}_evaluation.csv"), index=False)
        logging.info("\tDone.")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
