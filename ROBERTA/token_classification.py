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



class TokenClassificationDataset(Dataset):
    """Class for token classification dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None):
        super(TokenClassificationDataset, self).__init__(task_name, dataset_name, dataset_dir, url)

    def _fetch_dataset(self):
        """Fetch token classification dataset."""
        if not os.path.exists(self.dataset_dir):
            check_folder(self.dataset_dir)
            if self.dataset_name=='conll2003':
                self.url = self.url if self.url else 'https://raw.githubusercontent.com/kyzhouhzau/BERT-NER/master/data/'
            try:
                for _file in ['train.txt', 'dev.txt', 'test.txt']:
                    local_path = self.dataset_dir if self.dataset_dir else "./{}".format(self.dataset_name)
                    wget.download(os.path.join(self.url, _file), local_path)
            except Exception:
                raise FileNotFoundError("Invalid URL.")
    
    def process_dataset(self, set_type):
        if self.dataset_name=='conll2003':
            self.process_conll2003(set_type)

    def process_conll2003(self, set_type):
        """Process CoNLL2003 dataset.
        Be careful that the last line of your train/dev/test files is an empty line."""
        _file = set_type + '.txt'
        f = open(os.path.join(self.dataset_dir, _file))
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            if self.task=='ner':
                label.append(splits[-1][:-1])
            elif self.task=='pos-tagging':
                label.append(splits[1])

        if len(sentence) >0:
            data.append((sentence,label))
            sentence = []
            label = []
        if set_type=='train':
            self.train = data
        elif set_type=='test':
            self.test = data
        elif set_type=='dev':
            self.dev = data
    
    def get_labels(self):
        """ Returns possible labels for the task.
        """
        if not self.train:
            raise AttributeError("You need to process dataset before retrieving labels.")
        df = pd.DataFrame(self.train, columns =['sentence', 'labels'])
        all_lists = df.labels.values.tolist()
        all_values = [item for sublist in all_lists for item in sublist]
        labels = list(set(all_values)) + ["[CLS]", "[SEP]"]
        return sorted(labels)


class TokenClassificationProcessor(DataProcessor):
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
        for i, (sentence,labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = labels
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
            labellist = example.label
            encoded_dict = tokenizer.encode_plus(
                                example.text_a,                 # Sentence to encode.
                                add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                                max_length = max_seq_length,    # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt'           # Return pytorch tensors.
                        )            
            label_ids= []
            output_mask = []
            assert len(example.text_a.split(' '))==len(labellist)
            for i, word in enumerate(example.text_a.split(' ')):
                # duplicate labels for word pieces
                for j, _ in enumerate(tokenizer.tokenize(word)):
                    label_ids.append(label_map[labellist[i]])
                    output_mask.append(int((j == 0))) # 1 for first word piece of each word
            label_ids.insert(0, label_map['[CLS]'])
            output_mask.insert(0,1)
            label_ids = label_ids[0:(max_seq_length - 1)]
            output_mask = output_mask[0:(max_seq_length - 1)]
            label_ids.append(label_map['[SEP]'])
            output_mask.append(1)
            while len(label_ids)<encoded_dict['input_ids'].shape[-1]:
                label_ids.append(0)
                output_mask.append(0)
        
            features.append(InputFeatures(input_ids=encoded_dict['input_ids'],
                                    attention_mask=encoded_dict['attention_mask'],
                                    token_type_ids=encoded_dict['token_type_ids'],
                                    label_ids=torch.tensor(label_ids).unsqueeze(0),
                                    output_mask=torch.tensor(output_mask).unsqueeze(0)
                                    )
                            )
        return features
    
    def get_data_loader(self, features, batch_size, local_rank, set_type):
        """See base class."""
        input_ids = torch.cat([f.input_ids for f in features], dim=0)
        attention_mask = torch.cat([f.attention_mask for f in features], dim=0)
        token_type_ids =  torch.cat([f.token_type_ids for f in features], dim=0)
        label_ids =  torch.cat([f.label_ids for f in features], dim=0)
        output_mask =  torch.cat([f.output_mask for f in features], dim=0)
        data = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids, output_mask)
        if set_type=='train':
            if local_rank == -1:
                sampler = RandomSampler(data)
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader