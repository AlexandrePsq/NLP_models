"""
Class implementing GloVe embeddings.
"""

import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from tokenizer import tokenize

from utils import filter_args



class Glove(object):
    """Container for Glove model."""

    def __init__(self, pretrained_glove=None, language='english', embedding_size=None, training_set=None, **kwargs):
        super().__init__()
        if pretrained_glove is None:
            pretrained_glove = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/glove_training/glove.6B.300d.txt'
            embedding_size = 300
            training_set = 'Wikipedia-2014+Gigaword-5'
        self.init_embeddings(pretrained_glove, language)
        self.param = {'model_type':'GLOVE', 'embedding_size':embedding_size, 'training_set':training_set, 'language':language}
        self.param.update(**kwargs)
        self.update_model()
    
    def init_embeddings(self, path, language):
        """ Initialize an instance of Dictionary, which create
        (or retrieve) the dictionary associated with the data used.
        """
        embeddings_dict = {}
        with open(path, 'r', encoding="utf-8") as f: 
            for line in f: 
                values = line.split() 
                word = values[0] 
                vector = np.asarray(values[1:], "float32") 
                embeddings_dict[word] = vector 
        self.model = embeddings_dict
    
    def update_model(self):
        self.words2add = {'hadn':(['had', 'n’t'], 1),
                            'crossly':(['across'], 0), 
                            'mustn':(['must', 'n’t'], 1)} # the second value in the tuple is the number of following words to skip in generate
        for key in self.words2add.keys():
            self.model[key] = np.zeros((300,))
            for word in self.words2add[key][0]:
                self.model[key] += self.model[word]
            self.model[key] = self.model[key] / len(self.words2add[key][0])

    def __name__(self):
        """ Define the name of the instance of Glove using
        its arguments.
        """
        return '_'.join([self.param['model_type'], 'embedding-size', str(self.param['embedding_size']), 'language', self.param['language']])

    def generate(self, iterator, functions):
        dataframes = [function(self.model, iterator, **filter_args(function, self.param)) for function in functions]
        result = pd.concat([df for df in dataframes], axis = 1)
        return result

    