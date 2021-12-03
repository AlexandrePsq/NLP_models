""" 
"""

import os
import glob
import wget
import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from dataset import Dataset, InputExample, InputFeatures
from processors import DataProcessor
from bert_utils import check_folder
from joblib import Parallel, delayed


class MLMDataset(Dataset):
    """Class for MLM dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None, output_dir='./'):
        super(MLMDataset, self).__init__(task_name, dataset_name, dataset_dir, url)
        self.output_dir = output_dir

    def _fetch_dataset(self):
        """Fetch MLM dataset."""
        assert os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}train.txt'))
        assert os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}test.txt'))
        assert os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}dev.txt'))
    
    def process_dataset(self, set_type):
        self.process_mlm_dataset(set_type)

    def process_mlm_dataset(self, set_type):
        """Process CoNLL2003 dataset.
        Be careful that the last line of your train/dev/test files is an empty line."""
        if not os.path.exists(os.path.join(self.output_dir, 'train_examples.pkl')):
            self.train = open(os.path.join(self.dataset_dir, f'{self.dataset_name}train.txt'), 'r').read().lower().split(' \n ')
        if not os.path.exists(os.path.join(self.output_dir, 'test_examples.pkl')):
            self.test = open(os.path.join(self.dataset_dir, f'{self.dataset_name}test.txt'), 'r').read().lower().split(' \n ')
        if not os.path.exists(os.path.join(self.output_dir, 'dev_examples.pkl')):
            self.dev = open(os.path.join(self.dataset_dir, f'{self.dataset_name}dev.txt'), 'r').read().lower().split(' \n ')
            
    def get_labels(self):
        """ Returns possible labels for the task.
        """
        return []


class MLMProcessor(DataProcessor):
    """Processor for the MLM data set."""
              
    def __init__(self, max_seq_length, masking_proportion=15, device='cpu', output_dir='./'):
        self.max_seq_length = max_seq_length
        self.masking_proportion = masking_proportion
        self.device = device
        self.output_dir = output_dir

    def get_train_examples(self, dataset_object):
        """See base class."""
        if os.path.exists(os.path.join(self.output_dir, 'train_features.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'train_features.pkl'))
        elif os.path.exists(os.path.join(self.output_dir, 'train_examples.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'train_examples.pkl'))
        else:
            examples = self._create_examples(dataset_object.train, "train")
        return examples

    def get_dev_examples(self, dataset_object):
        """See base class."""
        if os.path.exists(os.path.join(self.output_dir, 'dev_features.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'dev_features.pkl'))
        elif os.path.exists(os.path.join(self.output_dir, 'dev_examples.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'dev_examples.pkl'))
        else:
            examples = self._create_examples(dataset_object.dev, "dev")
        return examples

    def get_test_examples(self, dataset_object):
        """See base class."""
        if os.path.exists(os.path.join(self.output_dir, 'test_features.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'test_features.pkl'))
        elif os.path.exists(os.path.join(self.output_dir, 'test_examples.pkl')):
            examples = self.load_object(os.path.join(self.output_dir, 'test_examples.pkl'))
        else:
            examples = self._create_examples(dataset_object.test, "test")
        return examples

    def mask_tokens(self, sequence):
        """Mask a given proportion of sequence tokens."""
        n_tokens = len(sequence)
        n_masked_tokens = int(self.masking_proportion*n_tokens/100)
        indexes = [random.randint(0, n_tokens-1) for i in range(n_masked_tokens)]
        while len(set(indexes))!=n_masked_tokens:
              indexes = [random.randint(0, n_tokens-1) for i in range(n_masked_tokens)]
        sequence = np.array(sequence)
        sequence[indexes] = 4
        return list(sequence)
              
    def pad_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        sequence = sequence[:self.max_seq_length]
        n = len(sequence)
        #return sequence + ['[PAD]'] * (self.max_seq_length - n)
        return sequence + [0] *(self.max_seq_length - n)

    def save_object(self, filename, data):
        """Save computed examples and features.
        """
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    
    def load_object(self, filename):
        """Load computed examples and features.
        """
        with open(filename, 'rb') as inp:  # Overwrites any existing file.
            data = pickle.load(inp)
        return data
        
    def _create_examples(self, lines, set_type):
        """Returns list of InputExample objects."""
        # Parallelizing a bit batch computation because it is quite slow...
        #lines = lines[:500]
        step = 18 # 17 sentences per input sequence
        #encoded_dict = self.tokenizer.encode('[CLS] ' + ' [SEP] [CLS] '.join(lines) + ' [SEP]')
        #tokens = np.array(encoded_dict.tokens)
        #ids = np.array(encoded_dict.ids)
        
        n = len(lines)
        
        def f(i, sequence):
            guid = "%s-%s" % (set_type, i)
            text_a = self.pad_to_max_length([2] + self.mask_tokens(sequence) + [3])
            text_b = None
            label = self.pad_to_max_length([2] + sequence + [3])
            example = InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label)
            return example
        
        def g(i, line):
            sequence = self.tokenizer.encode(' '.join(line)).ids
            return f(i, sequence)
        
        # Splitting data for memory issues...
        indexes = list(range(0, n, step))
        m = len(indexes)
        n_splits = 5
        splits = [indexes[i*m//n_splits: m*(i+1)//n_splits] for i in range(n_splits)]
        for index_split, split in enumerate(splits):
            print(f"Computing split {index_split+1} / {n_splits}... Split size: {len(split)}")
            examples = Parallel(n_jobs=-1)(delayed(g)(index+split[0], lines[i:i + step]) for index, i in tqdm(enumerate(split)))
            self.save_object(os.path.join(self.output_dir, f'{set_type}_examples_split-{index_split}.pkl'), examples)
        # Merging
        examples = [self.load_object(os.path.join(self.output_dir, f'{set_type}_examples_split-{index_split}.pkl')) for index_split in range(n_splits)]
        examples = [item for l in examples for item in l]
        self.save_object(os.path.join(self.output_dir, f'{set_type}_examples.pkl'), examples)
        return examples
              
    def set_tokenizer(self, tokenizer):
        """Set processor tokenizer."""
        self.tokenizer = tokenizer
              
    def batchify(self, i, iterator):
        """Batchify list of sentences."""
        print(f'Starting Batch {i}')
        iterator = [item.strip() for item in iterator]
        max_length = self.max_seq_length - 2 # for special tokens

        batches = []
        n = len(iterator)
        sentence_count = 0
        index_start = 0
        index_stop = 0

        while index_stop < n:
            if (len(self.tokenizer.encode(' '.join(iterator[index_start:index_stop+1])).tokens) < max_length):
                index_start += 1
                index_stop += 1
            while (len(self.tokenizer.encode(' '.join(iterator[index_start:index_stop+1])).tokens) < max_length) and (index_stop<n):
                index_stop += 1
            batches.append(iterator[index_start:index_stop])
            index_start = index_stop
        print(f'Batch {i} Done')
        return batches
    
    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer, set_type):
        """Loads a data file into a list of `InputBatch`s.
        Arguments:
            - label_list is discarded
        Returns:
            - input_ids: ids of ntokens + padding.
                e.g.: [103, 1023, 6423, 896, 102, 0, 0]
            - attention_mask: mask, 1 for tokens and 0 for padding.
                e.g.: [1, 1, 1, 1, 1, 0, 0]
            - token_type_ids: vector of 0.
                e.g.:[0, 0, 0, 0, 0, 0, 0]
            - label_ids: ids of the labels (there is 1 label for each word
            piece) + 0-padding
                e.g.: [1, 4, 4, 5, 2, 0, 0]
        """
        
        if os.path.exists(os.path.join(self.output_dir, f'{set_type}_features.pkl')):
            features = examples
        else:

            def f(example):
                labels_ids = torch.FloatTensor(example.label).unsqueeze(0).to(torch.int64)
                input_ids = torch.FloatTensor(example.text_a).unsqueeze(0).to(torch.int64)
                attention_mask = torch.ones(input_ids.size()).to(torch.int64)
                token_type_ids = torch.zeros(input_ids.size()).to(torch.int64)
                return InputFeatures(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        label_ids=labels_ids,
                                        output_mask=None)

            features = Parallel(n_jobs=-1)(delayed(f)(example) for example in tqdm(examples))
            self.save_object(os.path.join(self.output_dir, f'{set_type}_features.pkl'), features)

        return features
    
    def get_data_loader(self, features, batch_size, local_rank, set_type):
        """See base class."""
        input_ids = torch.cat([f.input_ids for f in features], dim=0)
        attention_mask = torch.cat([f.attention_mask for f in features], dim=0)
        token_type_ids =  torch.cat([f.token_type_ids for f in features], dim=0)
        label_ids =  torch.cat([f.label_ids for f in features], dim=0)
        data = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)
        if set_type=='train':
            if local_rank == -1:
                sampler = RandomSampler(data)
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader