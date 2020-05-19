"""
Glove Extractor
"""

import os

from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
from tokenizer import tokenize
from modeling_hacked_glove import Glove

from utils import embeddings



class GloveExtractor(object):
    """Container module for Glove embeddings extraction."""

    def __init__(self, config_path, language, name='', prediction_type='sequential', output_hidden_states=False):
        super().__init__()
        
        self.model = Glove()
        self.tokenizer = tokenize
        
        self.language = language
        self.FEATURE_COUNT = self.model.param['embedding-size']
        self.name = self.model.__name__()
        self.config = self.model.param
        self.prediction_type = prediction_type

    def __name__(self):
        """ Retrieve Glove instance name.
        """
        return self.model.__name__()

    def extract_activations(self, iterator, functions=[embeddings]):
        """ Extract embeddings from the model for each word from the input.
        Arguments: 
            - iterator: iterator object, 
            generally: iterator = tokenize(path, language, self.vocab)
        Returns:
            - result: pd.DataFrame containing activation
        """
        return self.model.generate(iterator, functions)
        