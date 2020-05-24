""" 
"""

import os
import wget
import torch
import glob
from tqdm import tqdm
import pandas as pd

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from dataset import Dataset, InputExample, InputFeatures
from processors import DataProcessor
from utils import check_folder, set_seed
from tokenizer import tokenize



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

    def get_train_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.train, "train")

    def get_dev_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.dev, "dev")

    def get_test_examples(self, dataset_object):
        """See base class."""
        return self._create_examples(dataset_object.test, "test")

    def _create_examples(self, lines, set_type):
        """Returns list of InputExample objects."""
        examples = []
        for i, (sentence, _) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = None
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
    
    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s.  
        """
        features = []

        for example in tqdm(examples):
            encoded_dict = tokenizer.encode_plus(
                                example.text_a,                 # Sentence to encode.
                                add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                                max_length = max_seq_length,    # Pad & truncate all sentences.
                                pad_to_max_length = False,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt'           # Return pytorch tensors.
                        )            

            labels = encoded_dict['input_ids'].clone()
            while encoded_dict['input_ids'].shape[-1] < max_seq_length:
                encoded_dict['input_ids'] = torch.cat([encoded_dict['input_ids'], torch.ones((1,1), dtype=torch.long)*tokenizer.encode('<|endoftext|>')[0]], dim=1)         
                encoded_dict['attention_mask'] = torch.cat([encoded_dict['attention_mask'], torch.zeros((1,1), dtype=torch.long)], dim=1)
                encoded_dict['token_type_ids'] = torch.cat([encoded_dict['token_type_ids'], torch.zeros((1,1), dtype=torch.long)], dim=1)
                labels = torch.cat([labels, torch.ones((1,1), dtype=torch.long)*(-100)], dim=1)
            features.append(InputFeatures(input_ids=encoded_dict['input_ids'],
                                    attention_mask=encoded_dict['attention_mask'],
                                    token_type_ids=encoded_dict['token_type_ids'],
                                    label_ids=labels
                                    )
                            )
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