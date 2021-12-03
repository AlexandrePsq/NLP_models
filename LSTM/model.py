"""
General Language Model based on recurrent neural network models
with the following architecture:
    Encoder -> RNN -> Decoder
The RNN model can implement either:
    - a GRU
    - or an LSTM
"""

import os

import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
from data import Corpus, Dictionary
from tokenizer import tokenize
from modeling_hacked_lstm import RNNModel
import lstm_utils as utils



class LSTMExtractor(object):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, config_path, language, name='', prediction_type='sequential', output_hidden_states=False, memory_size=np.inf, randomize=False, seed=1111):
        super().__init__()
        
        assert memory_size > 0
        utils.set_seed(seed)
        if randomize:
            parameters = utils.read_yaml(config_path)
            self.model = RNNModel(output_hidden_states=output_hidden_states, **parameters)
            if output_hidden_states:
                self.model.rnn.forward = lambda input, hidden: utils.forward(self.model.rnn, input, hidden, self.model.param)
        else:
            parameters = utils.read_yaml(config_path) if isinstance(config_path, str) else {'cuda': config_path['cuda']}
            device = torch.device("cuda" if parameters["cuda"] else "cpu")
            self.model = RNNModel.from_pretrained(config_path, output_hidden_states=output_hidden_states, device=device)
        self.tokenizer = tokenize
        
        self.language = language
        self.memory_size = memory_size
        self.NUM_HIDDEN_LAYERS = self.model.param['nlayers']
        self.FEATURE_COUNT = self.model.param['nhid']
        self.name = self.model.__name__()
        self.config = self.model.param
        self.prediction_type = prediction_type

    def __name__(self):
        """ Retrieve RNN instance name.
        """
        return self.model.name

    def extract_activations(self, iterator, language):
        """ Extract hidden state activations of the model for each word from the input.
        Optionally includes surprisal and entropy.
        Arguments: 
            - iterator: iterator object, 
            generally: iterator = tokenize(path, language, self.vocab)
            - includ_surprisal: bool specifying if we include surprisal
            - includ_entropy: bool specifying if we include entropy
            - parameters: list (of string representing gate names)
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        """
        self.model.eval()
        parameters = sorted(self.config['parameters'])
        columns_activations = ['{}-layer-{}-{}'.format(name, layer, i) for name in parameters for layer in range(1, 1 + self.NUM_HIDDEN_LAYERS) for i in range(1, 1 + self.FEATURE_COUNT)]
        activations = []
        surprisals = []
        entropies = []
        # Initialiazing variables
        out = None
        correction = 0
        
        final_iterator = iterator if self.memory_size==np.inf else utils.batchify_text_with_memory_size(iterator, self.memory_size)

        for index, item in tqdm(enumerate(final_iterator)):
            
            index += correction # correcting for <eos> symbols that should not be retrieved

            if index % self.memory_size == 0:
                hidden = self.model.init_hidden(1)
                inp = torch.autograd.Variable(torch.LongTensor([[self.model.vocab.word2idx[self.config['eos_separator']]]]))
                if self.config['cuda']:
                    inp = inp.cuda()
                # Start extracting activations
                out, hidden = self.model(inp, hidden)

            activation, surprisal, entropy, (out, hidden) = self.model.extract(item, out=out, hidden=hidden, parameters=parameters)

            if ((index + 1) % self.memory_size == 0) or (self.memory_size==np.inf) or (index < self.memory_size): # +1 because we look if the hidden state is reset at the next word
                if item=='<eos>':
                    correction -= 1
                else:
                    activations.append(activation)
                    surprisals.append(surprisal)
                    entropies.append(entropy)

        activations_df = pd.DataFrame(np.vstack(activations), columns=columns_activations)
        surprisals_df = pd.DataFrame(np.vstack(surprisals), columns=['surprisal'])
        entropies_df = pd.DataFrame(np.vstack(entropies), columns=['entropy'])
        result = pd.concat([activations_df, surprisals_df], axis = 1) if self.config['includ_surprisal'] else activations_df
        result = pd.concat([result, entropies_df], axis = 1) if self.config['includ_entropy'] else result
        return result
    
    