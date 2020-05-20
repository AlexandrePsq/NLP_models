"""
Class implementing GloVe embeddings.
"""

import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import filter_args



class BasicFeatures(object):
    """Container for Basic Features models."""

    def __init__(self, language='english', functions=[], **kwargs):
        super().__init__()
        self.param = {'language':language}
        self.param.update(**kwargs)
        self.functions = functions

    def __name__(self):
        """ Define the name of the instance of RNNModel using
        its arguments.
        """
        return '_'.join([function.__name__ for function in self.functions])

    def generate(self, iterator=None, functions=[], rms_iterator=None):
        dataframes = [function(iterator, **filter_args(function, self.param)) for function in functions if function.__name__!= 'rms']
        for function in functions:
            if function.__name__=='rms':
                dataframes.append(function(rms_iterator, **filter_args(function, self.param)))
        result = pd.concat([df for df in dataframes], axis = 1)
        return result

    