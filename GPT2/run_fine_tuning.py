""" Code to fine-tune hugging-face implementation of GPT2 model.
https://huggingface.co/
"""
import warnings
warnings.simplefilter(action='ignore')

import os
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

from tokenizers import ByteLevelBPETokenizer
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
    save_yaml(parameters, os.path.join(parameters['output_dir'], 'config.yml'))
    logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)
    logging.info("Parameters fetched.")

    logging.info("Setting seed for reproductibility...") 
    set_seed(parameters['seed'])
    logging.info("\tDone.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters['task'].lower()
    logging.info("\tDone.")

    logging.info("Instanciating dataset and data processor...")
    if task in ['language-modeling']:
        data = LMDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'])
        processor = LMProcessor(parameters['max_length'], device=device, output_dir=parameters['output_dir'])
    logging.info("\tDone.")

    logging.info("Fetching data (training + validation) and parameters...")
    data._fetch_dataset()
    for set_type in ['train', 'dev']:
        data.process_dataset(set_type)
    if parameters['do_test']:
        data.process_dataset('test')
    logging.info("\tDone.")

    logging.info("Fetching pre-trained GPT-2 model: {} and Tokenizer: {} for the task: {}...".format(parameters['pretrained_model'],parameters['pretrained_tokenizer'],parameters['task']))
    if task in ['language-modeling']:
        if parameters['start_from_scratch']:
            params = read_yaml(parameters['config_path'])
            params['layer_norm_epsilon'] = float(params['layer_norm_epsilon'])
            model = GPT2LMHeadModel(GPT2Config(**params))
        else:
            model = GPT2LMHeadModel.from_pretrained(
                        parameters['pretrained_model'],
                        output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                        output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
    if parameters['tokenizer_from_scratch']:
        tokenizer = ByteLevelBPETokenizer( 
                        lowercase=parameters['lowercase'])
        files = [os.path.join(parameters['dataset_dir'], item) for item in ['train.txt', 'test.txt', 'dev.txt']]
        tokenizer.train( 
                        files, 
                        vocab_size=parameters['vocab_size'], 
                        min_frequency=parameters['min_frequency'], 
                        show_progress=True, 
                        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        tokenizer.enable_truncation(max_length=512)
        tokenizer.save_model(parameters['output_dir'], parameters['dataset_name'] + 'tokenizer')
        processor.set_tokenizer(tokenizer)
        print(tokenizer.encode("<s> The dog ran <mask> outside . <unk> </s> <pad>").tokens) # --> ['<s>', 'Ġ', '<mask>', 'Ġ.', 'Ġ', '<unk>', 'Ġ', '</s>', 'Ġ', '<pad>']
        print(tokenizer.encode("<s> <mask> . <unk> </s> <pad>").ids) # --> [0, 225, 4, 272, 225, 3, 225, 2, 225, 1]
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(parameters['pretrained_tokenizer'])

    model.to(device)
    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info("\tDone.")

    logging.info("Get input examples...")
    train_examples = processor.get_train_examples(data)
    dev_examples = processor.get_dev_examples(data)
    if parameters['do_test']:
        test_examples = processor.get_test_examples(data)
    logging.info("\tDone.")

    logging.info("Get input features...")
    train_features = processor.convert_examples_to_features(train_examples, parameters['max_length'], tokenizer, set_type='train')
    dev_features = processor.convert_examples_to_features(dev_examples, parameters['max_length'], tokenizer, set_type='dev')
    if parameters['do_test']:
        test_features = processor.convert_examples_to_features(test_examples, parameters['max_length'], tokenizer, set_type='test')
    logging.info("\tDone.")
    
    logging.info("Creating data loaders...")
    train_dataloader = processor.get_data_loader(train_features, 
                                                    batch_size=parameters['batch_size'], 
                                                    local_rank=parameters['local_rank'], 
                                                    set_type='train')
    dev_dataloader = processor.get_data_loader(dev_features, 
                                                batch_size=parameters['batch_size'], 
                                                local_rank=parameters['local_rank'], 
                                                set_type='dev')
    if parameters['do_test']:
        test_dataloader = processor.get_data_loader(test_features, 
                                                    batch_size=parameters['batch_size'], 
                                                    local_rank=parameters['local_rank'], 
                                                    set_type='test')
    logging.info("\tDone.")

    logging.info("Creating optimizer and learning rate scheduler...")
    optimizer = AdamW(
                    model.parameters(),
                    lr=float(parameters['learning_rate']),
                    eps=float(parameters['adam_epsilon'])
                )
    total_steps = len(train_dataloader) * parameters['nb_epochs'] # Total number of training steps is [nb batches] x [nb epochs]. 
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=parameters['num_warmup_steps'],
                    num_training_steps=total_steps
                )
    logging.info("\tDone.")

    logging.info("Fine-tuning the model.")
    model_processor = ModelProcessor(model, optimizer, tokenizer, 
                                        scheduler, device, 
                                        parameters['metric_name'], 
                                        parameters['nb_epochs'],
                                        parameters['use_output_mask'])
    training_stats = model_processor.train(train_dataloader, dev_dataloader, parameters['output_dir'])
    logging.info("Validation reports: ")
    for stat in training_stats:
        logging.info(stat['report'])
    test_accuracy, test_loss = None, None
    if parameters['do_test']:
        logging.info("Evaluation report: ")
        test_accuracy, test_loss, test_time, report = model_processor.evaluate(test_dataloader) 
        logging.info(report)
    logging.info("\tDone.")

    logging.info("Saving fine-tuned model to {}...".format(os.path.join(parameters['output_dir'], 'fine_tuned')))
    save(model, tokenizer, parameters['output_dir'], 'fine_tuned')
    logging.info("\tDone.")

    logging.info("Plotting training and validation losses...")
    Report.plots_train_val_loss(training_stats, parameters['nb_epochs'], 
                                os.path.join(parameters['output_dir'], 'train_val_loss.png'), 
                                test_accuracy=test_accuracy, test_loss=test_loss)
    logging.info("\tDone.")
