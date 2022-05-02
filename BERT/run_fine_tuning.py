""" Code to fine-tune hugging-face implementation of BERT model.
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

from tokenizers import BertWordPieceTokenizer, Tokenizer
from transformers import BertForQuestionAnswering, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import BertForNextSentencePrediction, BertForSequenceClassification, BertForTokenClassification, BertForMaskedLM
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForMaskedLM, WEIGHTS_NAME, CONFIG_NAME

from sentence_classification import SentenceClassificationDataset, SentenceClassificationProcessor
from bert_utils import read_yaml, set_seed, format_time, filter_args, get_device, save, check_folder, save_yaml
from token_classification import TokenClassificationDataset, TokenClassificationProcessor
from mask_language_modeling import MLMDataset, MLMProcessor
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
    elif task == 'mask-language-modeling':
        data = MLMDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'], output_dir=parameters['output_dir'])
        processor = MLMProcessor(parameters['max_length'], parameters['masking_proportion'], device=device, output_dir=parameters['output_dir'], dataset_name=parameters['dataset_name'], dataset_dir=parameters['dataset_dir'])
    logging.info("\tDone.")

    logging.info("Fetching data (training + validation) and parameters...")
    data._fetch_dataset()
    for set_type in ['train', 'dev']:
        data.process_dataset(set_type)
    
    label_list = processor.get_labels(data)
    num_labels = len(label_list)
    if task in ['pos-tagging', 'ner']:
        num_labels += 1 # we add 1 because of the padding which is labelled 0
    logging.info("\tDone.")

    logging.info("Fetching pre-trained Bert model: {} and Tokenizer: {} for the task: {}...".format(parameters['pretrained_model'], parameters['pretrained_tokenizer'], parameters['task']))
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
    elif task == 'mask-language-modeling':
        if parameters['start_from_scratch']:
            params = read_yaml(parameters['config_path'])
            model = BertForMaskedLM(BertConfig(**params))
        else:
            model = BertForMaskedLM.from_pretrained(
                        parameters['pretrained_model'],
                        output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                        output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
    if parameters['tokenizer_from_scratch']:
        tokenizer = BertWordPieceTokenizer( 
                        clean_text=True, 
                        handle_chinese_chars=False, 
                        strip_accents=parameters['strip_accents'], 
                        lowercase=parameters['lowercase'])
        files = [os.path.join(parameters['dataset_dir'], item) for item in ['bert_train.txt', 'bert_test.txt', 'bert_dev.txt']]
        tokenizer.train( 
                        files, 
                        vocab_size=parameters['vocab_size'], 
                        min_frequency=parameters['min_frequency'], 
                        show_progress=True, 
                        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], 
                        limit_alphabet=parameters['limit_alphabet'], #1000 
                        wordpieces_prefix="##")
        tokenizer.enable_truncation(max_length=512)
        #tokenizer.save_model(os.path.join(parameters['output_dir'], parameters['dataset_name'] + 'tokenizer'))
        #tokenizer.save(os.path.join(parameters['output_dir'], parameters['dataset_name'] + 'tokenizer', 'tokenizer.json'))
        #tokenizer.save_pretrained(os.path.join(parameters['output_dir'], parameters['dataset_name'] + 'tokenizer'))
        print(tokenizer.encode("[CLS] [MASK] . [UNK] [SEP] [PAD]").ids) # --> [2, 4, 18, 1, 3, 0]
    else:
        tokenizer = Tokenizer.from_file(os.path.join(parameters['output_dir'], parameters['dataset_name'] + 'tokenizer', 'tokenizer.json'))

    processor.set_tokenizer(tokenizer)
    if parameters['start_epoch'] > 0:
        path = sorted(glob.glob(os.path.join(parameters['output_dir'], 'end-epoch-*')))
        if len(path)==0:
            path = sorted(glob.glob(os.path.join(parameters['output_dir'], 'start-epoch-*')))
        path = path[-1]
        print(f'Using model saved at: {path}...')
        model = BertForMaskedLM.from_pretrained(
                        path,
                        output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                        output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
        )
    model.to(device)
    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logging.info("\tDone.")

    logging.info("Get input examples...")
    train_examples_paths = processor.get_train_examples(data)
    dev_examples_paths = processor.get_dev_examples(data)
    logging.info("\tDone.")

    logging.info("Get input features...")
    train_features_paths = processor.convert_examples_to_features(train_examples_paths, label_list, parameters['max_length'], tokenizer, set_type='train')
    dev_features_paths = processor.convert_examples_to_features(dev_examples_paths, label_list, parameters['max_length'], tokenizer, set_type='dev')
    logging.info("\tDone.")
    
    logging.info("Creating optimizer and learning rate scheduler...")
    optimizer = AdamW(
                    model.parameters(),
                    lr=float(parameters['learning_rate']),
                    eps=float(parameters['adam_epsilon'])
                )
    nb_splits = 5
    nb_batches = nb_splits * len(processor.load_object(os.path.join(parameters['dataset_dir'], f"{parameters['dataset_name']}train_features_split-0.pkl")))
    total_steps = nb_batches * parameters['nb_epochs'] # Total number of training steps is [nb batches] x [nb epochs]. 
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=parameters['num_warmup_steps'],
                    num_training_steps=total_steps
                )
    logging.info("\tDone.")

    logging.info("Fine-tuning the model.")
    gc.collect()
    model_processor = ModelProcessor(model, optimizer, tokenizer, 
                                        scheduler, device, 
                                        parameters['metric_name'], 
                                        parameters['nb_epochs'],
                                        parameters['use_output_mask'])
    
    try:
        if parameters['do_train']:
            training_stats = model_processor.train(processor, train_features_paths, dev_features_paths, parameters['output_dir'], parameters=parameters)
            
            logging.info("Saving fine-tuned model to {}...".format(os.path.join(parameters['output_dir'], 'fine_tuned')))
            name = f"started_at_{parameters['init_checkpoints']}_fine_tuned" if parameters['init_checkpoints'] > 0 else 'fine_tuned'
            save(model_processor.model, tokenizer, parameters['output_dir'], name)
            logging.info("\tDone.")
        
        else:
            path = sorted(glob.glob(os.path.join(parameters['output_dir'], 'fine_tuned*')))[-1]
            print(f'Using model saved at: {path}...')
            model_processor.model = BertForMaskedLM.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
            model_processor.model.to(device)
            training_stats = pd.read_csv(os.path.join(parameters['output_dir'], 'training_stats.csv'))
            
    except KeyboardInterrupt:
        print('-' * 89)
        training_stats = pd.read_csv(os.path.join(parameters['output_dir'], 'training_stats.csv'))
        print('Exiting from training early')
    
    logging.info("Validation reports: ")
    for epoch, stat in training_stats.iterrows():
        logging.info(stat['report'])
    test_accuracy, test_loss = None, None
    
    if parameters['do_test']:
        data.process_dataset('test')
        test_examples_paths = processor.get_test_examples(data)
        test_features_paths = processor.convert_examples_to_features(test_examples_paths, label_list, parameters['max_length'], tokenizer, set_type='test')

        logging.info("Evaluation report: ")
        test_accuracy, test_loss, test_time, report = model_processor.evaluate(processor, test_features_paths, 'test', parameters) 
        testing_stats = [{
                    'Test. Loss': test_loss,
                    'Test. Accur.': test_accuracy,
                    'Test Time': test_time,
                    'report': report
                }]
        df = pd.DataFrame(data=testing_stats)
        df.to_csv(os.path.join(parameters['output_dir'], 'testing_stats.csv'), index=False)
        logging.info(df['report'].iloc[0])
    logging.info("\tDone.")
    
    logging.info("Plotting training and validation losses...")
    Report.plots_train_val_loss(training_stats, parameters['nb_epochs'], 
                                os.path.join(parameters['output_dir'], 'train_val_loss.png'), 
                                test_accuracy=test_accuracy, test_loss=test_loss)
    logging.info("\tDone.")
