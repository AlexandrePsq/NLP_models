""" 
"""
import warnings
warnings.simplefilter(action='ignore')

import os
import wget
import torch
import glob
import pickle
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

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None, language='english', extra=''):
        super(LMDataset, self).__init__(task_name, dataset_name, dataset_dir, url, extra=extra)
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
        if not os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}train_examples.pkl')):
            self.train = open(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}train.txt'), 'r').read().lower().split(' \n ')
        if not os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}test_examples.pkl')):
            self.test = open(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}test.txt'), 'r').read().lower().split(' \n ')
        if not os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}dev_examples.pkl')):
            self.dev = open(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}dev.txt'), 'r').read().lower().split(' \n ')
          

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
    
    def __init__(self, max_seq_length, device='cpu', output_dir='./', dataset_name='', dataset_dir='./', n_splits=5, context_size=None, extra=''):
        self.max_seq_length = max_seq_length if context_size is None else context_size+5 # +5 because of the special tokens + the current and following tokens
        print(f'Using context_size of: {context_size} and max_seq_length of {self.max_seq_length}')
        self.device = device
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.context_size=context_size
        self.extra = extra

    def get_data(self, dataset_object, set_type):
        """See base class."""
        if all([os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl')) for index_split in range(self.n_splits)]):
            paths = [os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl') for index_split in range(self.n_splits)]

        else:
            raise NotImplementedError()
        return paths
    
    def set_tokenizer(self, tokenizer):
        """Set processor tokenizer."""
        self.tokenizer = tokenizer
        
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
    
    def create_examples(self, sequence):
        """Returns list of InputExample objects."""
        #input_id = self.pad_to_max_length([0] + sequence + [1, 2]) ### HERE ###
        input_id = self.pad_to_max_length([50256] + sequence + [220, 50256])
        #attention_mask = self.pad_attention_to_max_length([1] + sequence + [1, 1])
        return input_id #, attention_mask

    def text_to_ids(self, lines, set_type):
        """Convert text file into list of ids associated to the tokenized objects."""
        # Parallelizing a bit batch computation because it is quite slow...
        #lines = lines[:5000]
        step = 18 # 17 sentences per input sequence
        n = len(lines)

        def g(i, line):
            sequence = self.tokenizer.encode(line).ids
            return f(i, sequence)
        
        # Splitting data for memory issues...
        n_splits = self.n_splits
        if self.context_size is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            print('Tokenizing...')
            for index_split in range(n_splits):
                start = index_split*len(lines)//n_splits 
                stop = (index_split+1)*len(lines)//n_splits
                all_ids = self.tokenizer.encode(' '.join(lines[start:stop])).ids
                self.save_object(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl'), all_ids)
                                
                #print(f'Computing examples split {index_split+1}...from {start} to {stop}... {len(all_ids)} ids')
                #examples = Parallel(n_jobs=-1)(delayed(f)(i, all_ids[i:i + self.context_size + 2]) for i, _ in tqdm(enumerate(all_ids[:-self.context_size -2])))
                ## +1 because we include the current token 
                ## and +1 because we want to predict the following token that has to be included...
                #print(f"Saving split {index_split+1} / {n_splits}... Split size: {len(examples)}")
                #self.save_object(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_examples_split-{index_split}.pkl'), examples)
            print('Computed.')
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
        else:
            indexes = list(range(0, n, step))
            m = len(indexes)
            splits = [indexes[i*m//n_splits: m*(i+1)//n_splits] for i in range(n_splits)]
            for index_split, split in enumerate(splits):
                print(f"Computing split {index_split+1} / {n_splits}... Split size: {len(split)}")
                examples = Parallel(n_jobs=-1)(delayed(g)(index+split[0], ' '.join(lines[i:i + step])) for index, i in tqdm(enumerate(split)))
                self.save_object(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_examples_split-{index_split}.pkl'), examples)
        # Merging
        #examples = [self.load_object(os.path.join(self.dataset_dir, f'{self.dataset_name}{set_type}_examples_split-{index_split}.pkl')) for index_split in range(n_splits)]
        #examples = [item for l in examples for item in l]
        #self.save_object(os.path.join(self.dataset_dir, f'{self.dataset_name}{set_type}_examples.pkl'), examples)
        
        examples_paths = [os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_examples_split-{index_split}.pkl') for index_split in range(n_splits)]
        
        return examples_paths
            
    def pad_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        n = len(sequence)
        if n==self.max_seq_length:
            return sequence
        else:
            print(f'Careful - {sequence} - is not of {len(sequence)} (!= max length)... Padding...')
            sequence = sequence[:self.max_seq_length]
            #result = sequence + [1] * ((self.max_seq_length - n)// 2) ### HERE ###
            result = sequence + [220, 50256] * ((self.max_seq_length - n)// 2)
            if len(result)==self.max_seq_length:
                return result
            else:
                #return result + [1] ### HERE ###
                return result + [220]
        
    def pad_attention_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        sequence = sequence[:self.max_seq_length]
        n = len(sequence)
        result = [1 for _ in sequence] + [0, 0] * ((self.max_seq_length - n)// 2)
        if len(result)==self.max_seq_length:
            return result
        else:
            return result + [0]
    
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
    
    def get_data_loader(self, features_path, batch_size, local_rank, set_type):
        """See base class."""
        # Loading features split
        if isinstance(features_path, str):
            features = self.load_object(features_path)
        else:
            features = [self.load_object(path) for path in features_path]
            features = [item for l in features for item in l]
        
        # Creating data loader
        input_ids = torch.cat([f.input_ids for f in features], dim=0)
        #attention_mask = None #torch.cat([f.attention_mask for f in features], dim=0)
        #token_type_ids =  torch.cat([f.token_type_ids for f in features], dim=0)
        #label_ids =  torch.cat([f.label_ids for f in features], dim=0)
        data = TensorDataset(input_ids) # attention_mask, token_type_ids, label_ids were removed !
        if set_type=='train':
            if local_rank == -1:
                sampler = RandomSampler(data)
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        #shuffle = (set_type=='train')
        #dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader
