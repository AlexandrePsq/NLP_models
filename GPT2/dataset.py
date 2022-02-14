"""
"""

import os



class Dataset(object):
    """Base class for dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None, extra=''):
        self.task = task_name
        self.dataset_name = dataset_name.lower()
        self.dataset_dir = dataset_dir
        self.url = url
        self.train = None
        self.test = None
        self.dev = None
        self.extra = extra

    def _fetch_dataset(self):
        """Fetch dataset."""
        raise NotImplementedError()

    def process_dataset(self,set_type):
        """Process dataset train/dev/test datasets.
        Be careful that the last line of your train/dev/test files is an empty line."""
        raise NotImplementedError()
    
    def get_labels(self):
        raise NotImplementedError()

    def read_file(self, set_type):
        if set_type=='train':
            data = self.train
        elif set_type=='test':
            data = self.test
        elif set_type=='dev':
            data = self.dev
        return data


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, output_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.output_mask = output_mask