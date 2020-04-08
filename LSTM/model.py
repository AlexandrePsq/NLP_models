"""
General Language Model based on recurrent neural network models
with the following architecture:
    Encoder -> RNN -> Decoder
The RNN model can implement either:
    - a GRU
    - or an LSTM
"""


import sys
import os

import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
from data import Corpus, Dictionary
from tokenizer import tokenize
import utils



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, eos_separator='<eos>', cuda=True):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        # hack the forward function to send an extra argument containing the model parameters
	    # self.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.backup = self.rnn.forward
        self.vocab = None
        self.cuda = cuda
        self.param = {'rnn_type':rnn_type, 'ntoken':ntoken, 'ninp':ninp, 
                        'nhid':nhid, 'nlayers':nlayers, 'dropout':dropout, 
                        'tie_weights':tie_weights, 'eos_separator': eos_separator}
    
    def init_vocab(self, path, language):
        """ Initialize an instance of Dictionary, which create
        (or retrieve) the dictionary associated with the data used.
        """
        self.vocab = Dictionary(path, language)

    def __name__(self):
        """ Define the name of the instance of RNNModel using
        its arguments.
        """
        return '_'.join([self.param['rnn_type'], 'embedding-size', str(self.param['ninp']),'nhid', str(self.param['nhid']), 'nlayers', str(self.param['nlayers']), 'dropout', str(self.param['dropout']).replace('.', '')])

    def init_weights(self):
        """ Initialize the weights of the model using 
        an uniform distribution and zero for the bias 
        of the decoder.
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        """ Concatenate the encoder, the recurrent neural 
        network model and the decoder.
        Arguments:
            - inp: torch.Variable (last predicted vector, or initializer token)
            - hidden: torch.Variable (last hidden state vector)
        Returns:
            - 
        """
        emb = self.drop(self.encoder(inp))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        """ Initialize to zeros the hidden state/cell.
        Arguments:
            - bsz: int, batch size
        Returns:
            - torch.Variable (or tuple of torch.Variable)
        """
        weight = next(self.parameters())
        if self.param['rnn_type'] == 'LSTM':
            return (weight.new_zeros(self.param['nlayers'], bsz, self.param['nhid']),
                    weight.new_zeros(self.param['nlayers'], bsz, self.param['nhid']))
        else:
            return weight.new_zeros(self.param['nlayers'], bsz, self.param['nhid'])

    def generate(self, iterator, includ_surprisal=False, includ_entropy=False, parameters=['in', 'forget', 'out', 'c_tilde', 'hidden', 'cell']):
        """ Extract hidden state activations of the model for each token from the input.
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
        parameters = sorted(parameters)
        # hack the forward function to send an extra argument containing the model parameters
        self.rnn.forward = lambda input, hidden: utils.forward(self.rnn, input, hidden, self.param)
        columns_activations = ['raw-{}-{}'.format(name, i) for name in parameters for i in range(self.param['nhid'] * self.param['nlayers'])]
        activations = []
        surprisals = []
        entropies = []
        # Initialiazing variables
        last_item = self.eos_separator
        out = None
        hidden = self.init_hidden(1)
        inp = torch.autograd.Variable(torch.LongTensor([[self.vocab.word2idx[self.eos_separator]]]))
        if self.cuda:
            inp = inp.cuda()
        # Start extracting activations
        out, hidden = self(inp, hidden)
        for item in tqdm(iterator):
            activation, surprisal, entropy, (out, hidden) = self.extract_activations(item, last_item=last_item, out=out, hidden=hidden, parameters=parameters)
            last_item = item
            activations.append(activation)
            surprisals.append(surprisal)
            entropies.append(entropy)
        activations_df = pd.DataFrame(np.vstack(activations), columns=columns_activations)
        surprisals_df = pd.DataFrame(np.vstack(surprisals), columns=['surprisal'])
        entropies_df = pd.DataFrame(np.vstack(entropies), columns=['entropy'])
        result = pd.concat([activations_df, surprisals_df], axis = 1) if includ_surprisal else activations_df
        result = pd.concat([result, entropies_df], axis = 1) if includ_entropy else result
        return result
    
    def extract_activations(self, item, last_item, out=None, hidden=None, parameters=['hidden']):
        """ Extract activations/surprisal/entropy for the processing of a given word.
        Arguments:
            - item: string (current real word)
            - last_item: string (last real word)
            - out: torch.Variable (last predicted output vector)
            - hidden: torch.Variable (last hidden state vector)
            - parameters: list (of string representing gate names)
        Returns:
            - activation: np.array (concatenated hidden state representation)
            - surprisal: np.array
            - entropy: np.array
            - (out, hidden): (torch.Variable, torch.Variable), (new predicted output, new hidden state)
        """
        activation = []
        # Surprisal is equal to the opposite of the probability that the correct word is 
        # predicted (+1)
        out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
        surprisal = -out[0, 0, self.vocab.word2idx[item]].item()
        # we extract the values of the parameters arguments while processing the real
        # word: item.
        inp = torch.autograd.Variable(torch.LongTensor([[self.vocab.word2idx[item]]]))
        if self.cuda:
            inp = inp.cuda()
        # The forward function has been hacked -> gates values and cell/hidden states 
        # are saved in self.rnn.gates.
        out, hidden = self(inp, hidden)
        pk = torch.nn.functional.softmax(out[0]).unsqueeze(0).detach().cpu().numpy()[0][0]
        # Entropy is a measure of the certainty of the network's prediction.
        entropy = -np.sum(pk * np.log2(pk), axis=0)
        for param in parameters:
            activation.append(self.rnn.gates[param].data.view(1,1,-1).cpu().numpy()[0][0])
        return np.hstack(activation), surprisal, entropy, (out, hidden)

