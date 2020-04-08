"""
General Language Model based on GPT-2 architecture.
The model can implement either:
    - GPT-2-small,
    - GPT-2-medium,
    - or a pre-trained GPT-2 architecture.
This class implements methods to retrieve hidden state or/and
attention heads activations.
"""


import sys
import os

import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
import utils
from tokenizer import tokenize 



class GPT2(object):
    """Container module for GPT-2."""

    def __init__(self, pretrained_gpt2_model, language, name, loi, cuda=False):
        super(GPT2, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        self.model = GPT2Model.from_pretrained(pretrained_gpt2_model, output_hidden_states=True, output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_gpt2_model)

        self.language = language
        self.NUM_HIDDEN_LAYERS = self.model.config['num_hidden_layers']
        self.FEATURE_COUNT = self.model.config['hidden_size']
        self.NUM_ATTENTION_HEADS = self.model.config['num_attention_heads']
        self.name = name
        self.loi = np.array(loi) if loi else np.arange(1 + self.NUM_HIDDEN_LAYERS) # loi: layers of interest
        self.cuda = cuda

    def __name__(self):
        """ Retrieve Bert instance name."""
        return self.name


    def generate(self, iterator, language, textgrid):
        """ Extract hidden state activations of the model for each token from the input, on a 
        word-by-word predictions or sentence-by-sentence prediction.
        Optionally includes surprisal and entropy.
        Input text should have one sentence per line, where each word and every 
        symbol is separated from the following by a space. No <eos> token should be included,
        as they are automatically integrated during tokenization.
        Arguments: 
            - iterator: iterator object, 
            generally: iterator = tokenize(path, language, self.vocab)
            - includ_surprisal: bool specifying if we include surprisal
            - includ_entropy: bool specifying if we include entropy
            - parameters: list (of string representing gate names)
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        iterator = tokenize(path, language, path_like=True, train=False)
        """
        activations = []
        self.model.eval()
        if self.cuda:
            self.model.to('cuda')
        for line in iterator:
            line = line.strip() # Remove trailing characters

            tokenized_text = self.tokenizer.tokenize(line)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda') if self.cuda else torch.tensor([indexed_tokens])

            with torch.no_grad():
                encoded_layers = self.model(tokens_tensor) # last_hidden_state, pooled_last_hidden_states, all_hidden_states
                # filtration
                if self.cuda:
                    encoded_layers = encoded_layers.to('cpu')
                encoded_layers = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                encoded_layers = encoded_layers[self.loi, :, :]
                activations += utils.extract_activations_from_tokenized(encoded_layers, mapping)
       
        result = pd.DataFrame(np.vstack(activations), columns=['layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        return result

