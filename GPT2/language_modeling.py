""" 
"""

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
        self.train = open(os.path.join(self.dataset_dir, f'{self.dataset_name}train.txt'), 'r').read().lower().split(' \n ')
        self.test = open(os.path.join(self.dataset_dir, f'{self.dataset_name}test.txt'), 'r').read().lower().split(' \n ')
        self.dev = open(os.path.join(self.dataset_dir, f'{self.dataset_name}dev.txt'), 'r').read().lower().split(' \n ')

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
    
    def __init__(self, max_seq_length, device='cpu', output_dir='./'):
        self.max_seq_length = max_seq_length
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

    def _create_examples(self, lines, set_type):
        """Returns list of InputExample objects."""
        # Parallelizing a bit batch computation because it is quite slow...
        #lines = lines[:5000]
        step = 18 # 17 sentences per input sequence
        n = len(lines)
        
        def f(i, sequence):
            guid = "%s-%s" % (set_type, i)
            text_a = self.pad_to_max_length([0] + sequence + [225, 2])
            text_b = None
            label = text_a
            example = InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label)
            return example
        
        def g(i, line):
            sequence = self.tokenizer.encode(' '.join(line)).ids
            return f(i, sequence)
        
        examples = Parallel(n_jobs=-1)(delayed(g)(index, lines[i:i + step]) for index, i in tqdm(enumerate(range(0, n, step))))
        self.save_object(os.path.join(self.output_dir, f'{set_type}_examples.pkl'), examples)
        
        return examples
    
    def pad_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        sequence = sequence[:self.max_seq_length]
        n = len(sequence)
        result = sequence + [225, 1] * ((self.max_seq_length - n)// 2)
        if len(result)==self.max_seq_length:
            return result
        else:
            return result + [225]
    
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
    
    def convert_examples_to_features(self, examples, max_seq_length, tokenizer, set_type):
        """Loads a data file into a list of `InputBatch`s.  
        """
        
        if os.path.exists(os.path.join(self.output_dir, f'{set_type}_features.pkl')):
            features = examples
        else:

            def f(example):
                labels_ids = torch.FloatTensor(example.label)[1:].unsqueeze(0).to(torch.int64)
                input_ids = torch.FloatTensor(example.text_a)[:-1].unsqueeze(0).to(torch.int64)
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