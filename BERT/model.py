"""
General Language Model based on BERT architecture.
The model can implement either:
    - BERT-base,
    - BERT-large,
    - or a pre-trained BERT architecture.
This class implements methods to retrieve hidden state or/and
attention heads activations.
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import utils
from modeling_hacked_bert import BertModel


class BertExtractor(object):
    """Container module for BERT."""

    def __init__(self, pretrained_bert_model, language, name, prediction_type, output_hidden_states, output_attentions, config_path=None):
        super(BertExtractor, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        self.model = BertModel.from_pretrained(pretrained_bert_model, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_model)
        
        self.language = language
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.config = utils.read_yaml(config_path) if config_path else {'max_length': 128}
        self.prediction_type = prediction_type # ['sentence', 'sequential']

    def __name__(self):
        """ Retrieve Bert instance name."""
        return self.name
    
    def extract_activations(self, iterator, language):
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
        """
        utils.set_seed()
        self.model.eval()
        if self.prediction_type == 'sentence':
            hidden_states_activations, attention_heads_activations = self.get_classic_activations(iterator, language)
        elif self.prediction_type == 'sequential':
            hidden_states_activations, attention_heads_activations = self.get_sequential_activations(iterator, language)
        return hidden_states_activations, attention_heads_activations 

    def get_classic_activations(self, iterator, language):
        """ Model predictions are generated in the classical way: the model
        take the whole sentence as input.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        # Here, we give as input the text line by line.
        for line in iterator:
            line = line.strip() # Remove trailing characters
            encoded_dict = self.tokenizer.encode_plus(
                                line,                               # Sentence to encode.
                                add_special_tokens = True,          # Add '[CLS]' and '[SEP]'
                                max_length = self.config['max_length'],   # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,       # Construct attn. masks.
                                return_tensors = 'pt'               # Return pytorch tensors.
                        )            
            # retrieve model inputs
            inputs_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']
            token_type_ids = encoded_dict['token_type_ids']

            line = '[CLS] ' + line + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(line)
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, line)

            with torch.no_grad():
                encoded_layers = self.model(inputs_ids, attention_mask, token_type_ids) # last_hidden_state, pooler_output, hidden_states, attentions
                # last_hidden_state dimension: (batch_size, sequence_length, hidden_size)
                # pooler_output dimension: (batch_size, hidden_size)
                # hidden_states dimension: num_layers * (batch_size, sequence_length, hidden_size)
                # attentions dimension: num_layers * (batch_size, num_heads, sequence_length, sequence_length)
                # hacked version: attentions dimension: num_layers * [(batch_size, sequence_length, hidden_size), 
                #                                                       (batch_size, num_heads, sequence_length, sequence_length)]
                # filtration
                if self.model.config.output_hidden_states:
                    hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping)
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([array[0].view([
                                                                1, 
                                                                self.config['max_length'], 
                                                                self.NUM_ATTENTION_HEADS, 
                                                                self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS]).permute(0, 2, 1, 3).contiguous()  for array in encoded_layers[3]])
                    attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping)
        hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return hidden_states_activations, attention_heads_activations
    
    def get_sequential_activations(self, iterator, language):
        """ Model predictions are generated sequentially: the model take as
        input the first word for the first prediction, then for each following
        prediction we add the next word for which we want to retrieve the activations.
        This framework enable to retrieve representations that are not aware of future
        tokens.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        # Here we give as input the sentence up to the actual word, incrementing by one at each step.
        for line in iterator:
            for index in range(1, len(line.split())):
                tmp_line = " ".join(line.split()[:index])
                tmp_line = tmp_line.strip() # Remove trailing characters
                encoded_dict = self.tokenizer.encode_plus(
                                    tmp_line,                           # Sentence to encode.
                                    add_special_tokens = True,          # Add '[CLS]' and '[SEP]'
                                    max_length = self.config['max_length'],   # Pad & truncate all sentences.
                                    pad_to_max_length = True,
                                    return_attention_mask = True,       # Construct attn. masks.
                                    return_tensors = 'pt'               # Return pytorch tensors.
                        )            
                # retrieve model inputs
                inputs_ids = encoded_dict['input_ids']
                attention_mask = encoded_dict['attention_mask']
                token_type_ids = encoded_dict['token_type_ids']

                tmp_line = '[CLS] ' + tmp_line + ' [SEP]'
                tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(tmp_line)
                mapping = utils.match_tokenized_to_untokenized(tokenized_text, tmp_line)

                with torch.no_grad():
                    encoded_layers = self.model(inputs_ids, attention_mask, token_type_ids) # dimension = layer_count * len(tokenized_text) * feature_count
                    # filtration
                    if self.model.config.output_hidden_states:
                        hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                        hidden_states_activations.append(utils.extract_activations_from_token_activations(hidden_states_activations_, mapping)[-1])
                    if self.model.config.output_attentions:
                        attention_heads_activations_ = np.vstack([array[0].view([
                                                                1, 
                                                                self.config['max_length'], 
                                                                self.NUM_ATTENTION_HEADS, 
                                                                self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS]).permute(0, 2, 1, 3).contiguous()  for array in encoded_layers[3]])
                        attention_heads_activations.append(utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping)[-1])
        hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(self.FEATURE_COUNT)])
        attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(self.NUM_ATTENTION_HEADS) for index in range(self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return hidden_states_activations, attention_heads_activations