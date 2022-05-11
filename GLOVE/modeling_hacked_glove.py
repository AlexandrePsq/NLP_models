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
        self.words2add_all = {'english': {'hadn':(['had', 'n’t'], 1),
                            'crossly':(['across'], 0), 
                            'mustn':(['must', 'n’t'], 1)}, # the second value in the tuple is the number of following words to skip in generate
                          'french': {'boa':(['serpent'], 0), 'avalait':(['avaler'], 0), 'fauve':(['animal'], 0), 'avalent':(['avaler'], 0), 'digérait':(['digérer'], 0), 'bélier':(['animal'], 0), 'taisais':(['taire'], 0), 'rirai':(['rire'], 0), 'riaient':(['rire'], 0), 'businessman':(['entrepreneur'], 0), 'baobab':(['arbre'], 0), 'riait':(['rire'], 0), '612':(['nombre'], 0), 'astéroïde':(['rocher'], 0), 'quitterai':(['quitter'], 0), 'écorce':(['bois'], 0), 'épines':(['pic'], 0), 'ramona':(['nettoyer'], 0), 'ramonés':(['nettoyer'], 0), 'ramoner':(['nettoyer'], 0), 'bâiller':(['fatigue'], 0), 'bâilla':(['fatigue'], 0), 'bâille':(['fatigue'], 0), 'bâillements':(['fatigue'], 0), 'grelots':(['cloche'], 0),'répéta':(['répéter'], 0),'aiguilleur':(['homme'], 0),'gronda':(['bruit'], 0),'grondant':(['bruit'], 0),'sahariens':(['désert'], 0),'sahara':(['désert'], 0),'poulie':(['outil'], 0),'répondis':(['répondre'], 0),'rougit':(['rouge'], 0),'monarque':(['roi'], 0),'hem':(['hum'], 0),'sire':(['homme'], 0),'vaniteux':(['prétentieux'], 0),'allumeur':(['homme'], 0),'réverbère':(['lumière'], 0),'éteignit':(['éteindre'], 0),'géographe':(['homme'], 0),'apprivoise':(['contrôler'], 0),'apprivoises':(['contrôler'], 0),'apprivoisé':(['contrôler'], 0),'boulon':(['outil'], 0),'sanglots':(['tristesse'], 0),'brindille':(['bois'], 0),'brindilles':(['bois'], 0),'obéissent':(['obéir'], 0),'obéissance':(['obéir'], 0),'obéissait':(['obéir'], 0),'obéi':(['obéir'], 0),'buveur':(['boire', 'homme'], 1),'explorateur':(['homme', 'aventure'], 1),'afrique':(['pays'], 0),'europe':(['pays'], 0),'russie':(['pays'], 0),
                          }}
        self.words2add = self.words2add_all[self.param['language']]
        for key in self.words2add.keys():
            if key not in self.model.keys():
                self.model[key] = np.zeros((self.param['embedding_size'],))
                for word in self.words2add[key][0]:
                    try:
                        self.model[key] += self.model[word]
                    except:
                        print(f'{word} does not appear in the vocabulary... Be sure that it is normal.')
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

    