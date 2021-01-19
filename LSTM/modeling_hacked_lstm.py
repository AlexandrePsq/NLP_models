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
import utils



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, eos_separator='<eos>', cuda=True, **kwargs):
        super().__init__()
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
        self.init_vocab(kwargs['path_to_vocab'], kwargs['language'])
        if not torch.cuda.is_available():
            cuda = False
        self.cuda = cuda
        self.eos_separator = eos_separator
        self.param = {'rnn_type':rnn_type, 'ntoken':ntoken, 'ninp':ninp, 
                        'nhid':nhid, 'nlayers':nlayers, 'dropout':dropout, 
                        'tie_weights':tie_weights, 'eos_separator': eos_separator,
                        'cuda': cuda}
        self.param.update(**kwargs)
    
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

    @classmethod
    def from_pretrained(cls, path, output_hidden_states=False):
        if os.path.exists(path):
            parameters = utils.read_yaml(path)
        else:
            raise ValueError("{} doesn't exists.".format(path))
        model = cls(**parameters)
        model.load_state_dict(torch.load(parameters['weights_path']))
        if output_hidden_states:
            model.rnn.forward = lambda input, hidden: utils.forward(model.rnn, input, hidden, model.param)
        return model
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def extract(self, item, out=None, hidden=None, parameters=['hidden']):
        """ Extract activations/surprisal/entropy for the processing of a given word.
        Arguments:
            - item: string (current real word)
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
        out = torch.nn.functional.log_softmax(out[0], dim=-1).unsqueeze(0)
        surprisal = -out[0, 0, self.vocab.word2idx[item]].item()
        # we extract the values of the parameters arguments while processing the real
        # word: item.
        inp = torch.autograd.Variable(torch.LongTensor([[self.vocab.word2idx[item]]]))
        if self.cuda:
            inp = inp.cuda()
        # The forward function has been hacked -> gates values and cell/hidden states 
        # are saved in self.rnn.gates.
        out, hidden = self(inp, hidden)
        pk = torch.nn.functional.softmax(out[0], dim=-1).unsqueeze(0).detach().cpu().numpy()[0][0]
        # Entropy is a measure of the certainty of the network's prediction.
        entropy = -np.sum(pk * np.log2(pk), axis=0)
        for param in parameters:
            activation.append(self.rnn.gates[param].data.view(1,1,-1).cpu().numpy()[0][0])
        return np.hstack(activation), surprisal, entropy, (out, hidden)

