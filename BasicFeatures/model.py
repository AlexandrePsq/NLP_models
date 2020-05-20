"""
Glove Extractor
"""

import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from tokenizer import tokenize
from modeling_hacked_basic_features import BasicFeatures

from utils import filter_args



class BasicFeaturesExtractor(object):
    """Container module for Basic Features extraction."""

    def __init__(self, basic_features, language='english', prediction_type='sequential', **kwargs):
        super().__init__()
        
        self.model = BasicFeatures(functions=basic_features, language=language, **kwargs)
        self.tokenizer = tokenize
        
        self.language = language
        self.FEATURE_COUNT = len(basic_features)
        self.name = self.model.__name__()
        self.config = self.model.param
        self.prediction_type = prediction_type

    def __name__(self):
        """ Retrieve Glove instance name.
        """
        return self.model.__name__()

    def extract_activations(self, iterator=None, rms_iterator=None):
        """ Extract embeddings from the model for each word from the input.
        Arguments: 
            - iterator: iterator object, 
            generally: iterator = tokenize(path, language, self.vocab)
        Returns:
            - result: pd.DataFrame containing activation
        """
        return self.model.generate(iterator, self.model.functions, rms_iterator)
        