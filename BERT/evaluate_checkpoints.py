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
    save_yaml(parameters, os.path.join(parameters['output_dir'], 'config_evaluation.yml'))
    logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)
    logging.info("Parameters fetched.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters['task'].lower()
    logging.info("\tDone.")

    logging.info("Instanciating dataset and data processor...")
    data = MLMDataset(task, parameters['dataset_name'].lower(), dataset_dir=parameters['dataset_dir'], output_dir=parameters['output_dir'])
    processor = MLMProcessor(parameters['max_length'], parameters['masking_proportion'], device=device, output_dir=parameters['output_dir'], dataset_name=parameters['dataset_name'], dataset_dir=parameters['dataset_dir'])
    logging.info("\tDone.")

    logging.info("Fetching data (training + validation) and parameters...")
    data._fetch_dataset()
    
    label_list = processor.get_labels(data)
    num_labels = len(label_list)
    logging.info("\tDone.")

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
    
    print(tokenizer.encode("[CLS] [MASK] . [UNK] [SEP] [PAD]").ids) # --> [2, 4, 12, 1, 3, 0]
    
    processor.set_tokenizer(tokenizer)
    paths = sorted(glob.glob(os.path.join(parameters['output_dir'], 'checkpoint_*')))

    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logging.info("\tDone.")

    logging.info("Creating optimizer and learning rate scheduler...")
    nb_splits = 5
    nb_batches = nb_splits * len(processor.load_object(os.path.join(parameters['dataset_dir'], f"{parameters['dataset_name']}train_features_split-0.pkl")))
    model_processor = ModelProcessor(None, None, None, 
                                        None, device, 
                                        parameters['metric_name'], 
                                        None,
                                        parameters['use_output_mask'])
    logging.info("\tDone.")

    gc.collect()
    data.process_dataset('test')
    test_examples_paths = processor.get_test_examples(data)
    test_features_paths = processor.convert_examples_to_features(test_examples_paths, label_list, parameters['max_length'], tokenizer, set_type='test')
    testing_stats  = []
    try:
        for path in paths:
            print(f'Using model saved at: {path}...')
            logging.info(f'Using model saved at: {path}...')
            logging.info("Setting seed for reproductibility...") 
            set_seed(parameters['seed'])
            logging.info("\tDone.")
            model_processor.model = BertForMaskedLM.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
            model_processor.model.to(device)
            test_accuracy, test_loss = None, None
            test_accuracy, test_loss, test_time, report = model_processor.evaluate(processor, test_features_paths, 'test', parameters) 
            testing_stats.append({
                        'Test. Loss': test_loss,
                        'Test. Accur.': test_accuracy,
                        'Test Time': test_time,
                        'model_path': path,
                        'report': report})
            df = pd.DataFrame(data=testing_stats)
            df.to_csv(os.path.join(parameters['output_dir'], 'testing_stats_checkpoints.csv'), index=False)
        df = pd.DataFrame(data=testing_stats)
        df.to_csv(os.path.join(parameters['output_dir'], 'testing_stats_checkpoints.csv'), index=False)
        logging.info("\tDone.")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
