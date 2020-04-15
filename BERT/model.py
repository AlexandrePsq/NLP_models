"""
General Language Model based on BERT architecture.
The model can implement either:
    - BERT-base,
    - BERT-large,
    - or a pre-trained BERT architecture.
This class implements methods to retrieve hidden state or/and
attention heads activations.
"""


import sys
import os

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import utils
from tokenizer import tokenize 



class BERT(object):
    """Container module for BERT."""

    def __init__(self, pretrained_bert_model, language, name, loi, prediction_level, output_hidden_states, output_attentions):
        super(BERT, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        self.model = BertModel.from_pretrained(pretrained_bert_model, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)
        
        self.language = language
        self.NUM_HIDDEN_LAYERS = self.model.config['num_hidden_layers']
        self.FEATURE_COUNT = self.model.config['hidden_size']
        self.NUM_ATTENTION_HEADS = self.model.config['num_attention_heads']
        self.name = name
        self.prediction_level = prediction_level
        self.loi = np.array(loi) if loi else np.arange(1 + self.NUM_HIDDEN_LAYERS) # loi: layers of interest

    def __name__(self):
        """ Retrieve Bert instance name."""
        return self.name


    def generate(self, iterator, language):
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
        # iterator = tokenize(path, language, path_like=True, train=False)
        """
        attentions_activations = []
        hidden_states_activations = []
        self.model.eval()
        if self.prediction_level == 'sentence':
            # Here, we give as input the text line by line.
            for line in iterator:
                line = line.strip() # Remove trailing characters

                line = '[CLS] ' + line + ' [SEP]'
                tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(line)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                segment_ids = [1 for x in tokenized_text]
                mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])

                with torch.no_grad():
                    encoded_layers = self.model(tokens_tensor, segments_tensors) # last_hidden_state, pooler_output, hidden_states, attentions
                    # last_hidden_state dimension: (batch_size, sequence_length, hidden_size)
                    # pooler_output dimension: (batch_size, hidden_size)
                    # hidden_states dimension: num_layers * (batch_size, sequence_length, hidden_size)
                    # attentions dimension: num_layers * (batch_size, num_heads, sequence_length, sequence_length)
                    # filtration
                    if self.model.config.output_hidden_states:
                        hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                        hidden_states_activations_ = hidden_states_activations_[self.loi, :, :]
                        hidden_states_activations += utils.extract_hidden_state_activations_from_tokenized(hidden_states_activations_, mapping)
                    if self.model.config.output_attentions:
                        attentions_activations_ = np.vstack(encoded_layers[3])
                        attentions_activations_ = attentions_activations_[self.loi, :, :]
                        attentions_activations += utils.extract_attention_head_activations_from_tokenized(attentions_activations_, mapping)
        elif self.prediction_level == 'sequential':
            # Here we give as input the sentence up to the actual word, incrementing by one at each step.
            for line in iterator:
                for index in range(1, len(line.split())):
                    tmp_line = " ".join(line.split()[:index])
                    tmp_line = tmp_line.strip() # Remove trailing characters

                    tmp_line = '[CLS] ' + tmp_line + ' [SEP]'
                    tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(tmp_line)
                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                    segment_ids = [1 for x in tokenized_text]
                    mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

                    # Convert inputs to PyTorch tensors
                    tokens_tensor = torch.tensor([indexed_tokens])
                    segments_tensors = torch.tensor([segment_ids])

                    with torch.no_grad():
                        encoded_layers = self.model(tokens_tensor, segments_tensors) # dimension = layer_count * len(tokenized_text) * feature_count
                        # filtration
                        if self.model.config.output_hidden_states:
                            hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                            hidden_states_activations_ = hidden_states_activations_[self.loi, :, :]
                            hidden_states_activations.append(utils.extract_hidden_state_activations_from_tokenized(hidden_states_activations_, mapping)[-1])
                        if self.model.config.output_attentions:
                            attentions_activations_ = np.vstack(encoded_layers[3])
                            attentions_activations_ = attentions_activations_[self.loi, :, :]
                            attentions_activations.append(utils.extract_attention_head_activations_from_tokenized(attentions_activations_, mapping)[-1])
        hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        attentions_activations = pd.DataFrame(np.vstack(attentions_activations), columns=['attention-layer-{}-{}'.format(layer, index) for layer in self.loi for index in range(self.FEATURE_COUNT)])
        return hidden_states_activations, attentions_activations