""" This training code is based on the `run_glue.py` script here:
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
"""


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

from transformers import BertForQuestionAnswering, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertForNextSentencePrediction, BertForSequenceClassification, BertForTokenClassification
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForMaskedLM, WEIGHTS_NAME, CONFIG_NAME

from sentence_classification import SentenceClassificationDataset, SentenceClassificationProcessor
from utils import read_yaml, set_seed, format_time, filter_args, get_device, save, check_folder, save_yaml
from token_classification import TokenClassificationDataset, TokenClassificationProcessor
from processors import DataProcessor, ModelProcessor
from reporting import Report
from dataset import Dataset



########################################################################################################
# ------------------------------------------- FINE - TUNING -------------------------------------------#
########################################################################################################

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Fine-tune a BERT model for a specific NLP task.")
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
    if task in ['pos-tagging', 'ner']:
        data = TokenClassificationDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'])
        processor = TokenClassificationProcessor()
    elif task == 'sentence-classification': # to be modified
        data = SentenceClassificationDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'])
        processor = SentenceClassificationProcessor()
    logging.info("\tDone.")

    logging.info("Fetching data (training + validation) and parameters...")
    data._fetch_dataset()
    for set_type in ['train', 'dev']:
        data.process_dataset(set_type)
    if parameters['do_test']:
        data.process_dataset('test')
    label_list = processor.get_labels(data)
    num_labels = len(label_list)
    if task in ['pos-tagging', 'ner']:
        num_labels += 1 # we add 1 because of the padding which is labelled 0
    logging.info("\tDone.")

    logging.info("Fetching pre-trained Bert model: {} and Tokenizer: {} for the task: {}...".format(parameters['pretrained_model'],
                                                                                                    parameters['pretrained_tokenizer'],
                                                                                                    parameters['task']))
    if task in ['pos-tagging', 'ner']:
        model = BertForTokenClassification.from_pretrained(
                    parameters['pretrained_model'],
                    num_labels=num_labels, # The number of output labels for classification.  
                    output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                    output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
        )
    elif task in ['sentence-classification']:
        model = BertForSequenceClassification.from_pretrained(
                    parameters['pretrained_model'],
                    num_labels=num_labels, # The number of output labels for classification.  
                    output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                    output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
        )
    tokenizer = BertTokenizer.from_pretrained(parameters['pretrained_tokenizer'])
    model.to(device)
    logging.info("\tDone.")

    logging.info("Get input examples...")
    train_examples = processor.get_train_examples(data)
    dev_examples = processor.get_dev_examples(data)
    if parameters['do_test']:
        test_examples = processor.get_test_examples(data)
    logging.info("\tDone.")

    logging.info("Get input features...")
    train_features = processor.convert_examples_to_features(train_examples, label_list, parameters['max_length'], tokenizer) 
    dev_features = processor.convert_examples_to_features(dev_examples, label_list, parameters['max_length'], tokenizer)
    if parameters['do_test']:
        test_features = processor.convert_examples_to_features(test_examples, label_list, parameters['max_length'], tokenizer) 
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
    logging.info("Reports durinf validations: ")
    for stat in training_stats:
        logging.info(stat['report'])
    test_accuracy, test_loss = None, None
    if parameters['do_test']:
        logging.info("Evaluation report: ")
        test_accuracy, test_loss, test_time, report = model_processor.evaluate(test_dataloader) 
        logging.info(report)
    logging.info("\tDone.")

    logging.info("Saving fine-tuned model...")
    save(model, tokenizer, parameters['output_dir'], 'fine_tuned')
    logging.info("\tDone.")

    logging.info("Plotting training and validation losses...")
    Report.plots_train_val_loss(training_stats, parameters['nb_epochs'], 
                                os.path.join(parameters['output_dir'], 'train_val_loss.png'), 
                                test_accuracy=test_accuracy, test_loss=test_loss)
    logging.info("\tDone.")
