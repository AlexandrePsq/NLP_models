""" 
"""

import os
import wget
import torch
import glob
import pandas as pd
from tqdm import tqdm
import multiprocessing

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from dataset import Dataset, InputExample, InputFeatures
from processors import DataProcessor
from gpt2_utils import check_folder, set_seed
from tokenizer import tokenize

from joblib import Parallel, delayed



class LMDataset(Dataset):
    """Class for language modeling dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None, language='english'):
        super(LMDataset, self).__init__(task_name, dataset_name, dataset_dir, url)
        self.language = language

    def _fetch_dataset(self):
        """Fetch sentence classification dataset."""
        if not os.path.exists(self.dataset_dir):
            check_folder(self.dataset_dir)
            if self.dataset_name=='lpp':
                pass
    
    def process_dataset(self, set_type):
        if self.dataset_name=='lpp':
            self.process_lpp(set_type)
        else:
            self.process_gutenberg(set_type)
            print(f"Using default Gutenberg dataset {set_type} {self.dataset_name}...")
    
    def process_gutenberg(self, set_type):
        """Process Gutenberg dataset.
        The result is an iterator of tuples (sentence, label)."""
        self.train = open(os.path.join(self.dataset_dir, f'{self.dataset_name}train.txt'), 'r').read().lower().split(' \n ')[:100]
        self.test = open(os.path.join(self.dataset_dir, f'{self.dataset_name}test.txt'), 'r').read().lower().split(' \n ')[:100]
        self.dev = open(os.path.join(self.dataset_dir, f'{self.dataset_name}dev.txt'), 'r').read().lower().split(' \n ')[:100]

    def process_lpp(self, set_type):
        """Process LPP dataset.
        The result is an iterator of tuples (sentence, label)."""
        set_seed()
        _file = 'text_english_run*.txt'
        path_to_data = os.path.join(self.dataset_dir, _file)
        files = sorted(glob.glob(path_to_data))
        iterator_list = [tokenize(path, self.language, train=False) for path in files]
        iterator = [item for sub_l in iterator_list for item in sub_l]
        sentences = iterator.copy()
        labels = [None] * len(sentences)
        data = zip(sentences, labels)
        if set_type=='train':
            self.train = list(data).copy()
        elif set_type=='test':
            self.test = list(data).copy()
        elif set_type=='dev':
            self.dev = list(data).copy()
    
    def get_labels(self):
        """ Returns possible labels for the task.
        """
        raise NotImplementedError()


class LMProcessor(DataProcessor):
    """Processor for language modeling."""
    
    def __init__(self, max_seq_length, device='cpu'):
        self.max_seq_length = max_seq_length
        self.device = device

    def get_train_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.train, "train")

    def get_dev_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.dev, "dev")

    def get_test_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.test, "test")
    
    def set_tokenizer(self, tokenizer):
        """Set processor tokenizer."""
        self.tokenizer = tokenizer

    def _create_examples(self, lines, set_type):
        """Returns list of InputExample objects."""
        # Parallelizing a bit batch computation because it is quite slow...
        lines = lines[:5000]
        step = 18 # 17 sentences per input sequence
        n = len(lines)
        
        def g(lines, i, step):
            return self.tokenizer.encode(' '.join(lines[i:i + step])).ids
        
        batches = Parallel(n_jobs=-1)(delayed(g)(lines, i, step) for i in tqdm(range(0, n, step))) #'<|endoftext|> '

        def f(i, sequence):
            guid = "%s-%s" % (set_type, i)
            text_a = self.pad_to_max_length([2] + sequence + [3])
            text_b = None
            label = text_a
            example = InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label)
            return example
        
        examples = Parallel(n_jobs=-1)(delayed(f)(i, sequence) for i, sequence in tqdm(enumerate(batches)))
        
        return examples
    
    def pad_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        sequence = sequence[:self.max_seq_length]
        n = len(sequence)
        #return sequence + ['[PAD]'] * (self.max_seq_length - n)
        return sequence + [0] *(self.max_seq_length - n)
    
    def batchify(self, iterator):
        """Batchify list of sentences."""
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
            batches.append(iterator[index_start:index_stop+1])
            index_start = index_stop

        return batches
    
    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s.  
        """
        
        def f(example):
            labels_ids = torch.FloatTensor(example.label).unsqueeze(0).to(torch.int64).to(self.device)[1:]
            input_ids = torch.FloatTensor(example.text_a).unsqueeze(0).to(torch.int64).to(self.device)[:-1]
            attention_mask = torch.ones(input_ids.size()).to(torch.int64).to(self.device)
            token_type_ids = torch.zeros(input_ids.size()).to(torch.int64).to(self.device)
            return InputFeatures(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    label_ids=labels_ids,
                                    output_mask=None)
        
        features = Parallel(n_jobs=-1)(delayed(f)(example) for example in tqdm(examples))
                                       
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
                sampler = RandomSampler(data, num_samples=len(data))
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader