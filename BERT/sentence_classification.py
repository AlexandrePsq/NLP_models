""" 
"""

import os
import wget
import torch
import pandas as pd

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from dataset import Dataset, InputExample, InputFeatures
from processors import DataProcessor
from utils import check_folder



class SentenceClassificationDataset(Dataset):
    """Class for sentence classification dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None):
        super(SentenceClassificationDataset, self).__init__(task_name, dataset_name, dataset_dir, url)

    def _fetch_dataset(self):
        """Fetch sentence classification dataset."""
        if not os.path.exists(self.dataset_dir):
            check_folder(self.dataset_dir)
            if self.dataset_name=='cola':
                self.url = self.url if self.url else 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
                try:
                    local_path = os.path.join(self.dataset_dir, 'cola.zip')
                    wget.download(self.url, local_path)
                    # Unzip the dataset (if we haven't already)
                    os.system("unzip {} -d {}".format(local_path, self.dataset_dir))
                    os.system("mv {} {}".format(os.path.join(self.dataset_dir, 'cola_public/raw/in_domain_dev.tsv'),
                                                os.path.join(self.dataset_dir, 'dev.tsv')))
                    os.system("mv {} {}".format(os.path.join(self.dataset_dir, 'cola_public/raw/in_domain_train.tsv'),
                                                os.path.join(self.dataset_dir, 'train.tsv')))
                    os.system("mv {} {}".format(os.path.join(self.dataset_dir, 'cola_public/raw/out_of_domain_dev.tsv'),
                                                os.path.join(self.dataset_dir, 'test.tsv')))
                    #os.system("rm {}".format(local_path))
                    #os.system("rm -r {}".format(os.path.join(self.dataset_dir, 'cola_public')))
                except Exception:
                    raise FileNotFoundError("Invalid URL.")
            if self.dataset_name=='sst-2':
                self.url = self.url if self.url else 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/'
                try:
                    for _file in ['train.tsv', 'dev.tsv', 'test.tsv']:
                        local_path = self.dataset_dir if self.dataset_dir else "./{}".format(self.dataset_name)
                        wget.download(os.path.join(self.url, _file), local_path)
                except Exception:
                    raise FileNotFoundError("Invalid URL.")
    
    def process_dataset(self, set_type):
        if self.dataset_name=='cola':
            self.process_cola(set_type)
        elif self.dataset_name=='sst-2':
            self.process_sst2(set_type)

    def process_cola(self, set_type):
        """Process CoLA dataset.
        The result is an iterator of tuples (sentence, label)."""
        _file = set_type + '.tsv'
        path_to_data = os.path.join(self.dataset_dir, _file)
        df = pd.read_csv(path_to_data, 
                            delimiter='\t', 
                            header=None, 
                            names=['sentence_source', 'label', 'label_notes', 'sentence'])
        sentences = df.sentence.values
        labels = df.label.values
        data = zip(sentences, labels)
        if set_type=='train':
            self.train = list(data).copy()
        elif set_type=='test':
            self.test = list(data).copy()
        elif set_type=='dev':
            self.dev = list(data).copy()
    
    def process_sst2(self, set_type):
        """Process SST-2 dataset.
        The result is an iterator of tuples (sentence, label)."""
        _file = set_type + '.tsv'
        path_to_data = os.path.join(self.dataset_dir, _file)
        df = pd.read_csv(path_to_data, 
                            delimiter='\t')
        sentences = df.sentence.values
        labels = df.label.values
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
        if not self.train:
            raise AttributeError("You need to process dataset before retrieving labels.")
        df = pd.DataFrame(self.train, columns =['sentence', 'labels'])
        labels = list(df.labels.unique())
        return sorted(labels)


class SentenceClassificationProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

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
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
    
    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s.
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
        features = []
        label_map = {label : i for i, label in enumerate(label_list,1)}

        for example in examples:
            encoded_dict = tokenizer.encode_plus(
                                example.text_a,                 # Sentence to encode.
                                add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                                max_length = max_seq_length,    # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt'           # Return pytorch tensors.
                        )            
            features.append(InputFeatures(input_ids=encoded_dict['input_ids'],
                                    attention_mask=encoded_dict['attention_mask'],
                                    token_type_ids=encoded_dict['token_type_ids'],
                                    label_ids=torch.tensor(example.label).unsqueeze(0)
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