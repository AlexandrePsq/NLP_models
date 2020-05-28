"""
General Language Model based on GPT2 architecture.
The model can implement any pre-trained GPT2 architecture.
This class implements methods to retrieve hidden state or/and
attention heads activations.
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import utils
from modeling_hacked_gpt2 import GPT2Model


class GPT2Extractor(object):
    """Container module for GPT2."""

    def __init__(self, pretrained_gpt2_model, language, name, prediction_type, output_hidden_states, output_attentions, config_path=None, max_length=512, context_length=250, number_of_sentence=1, number_sentence_before=0):
        super(GPT2Extractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_gpt2_model)
        self.model = GPT2Model.from_pretrained(pretrained_gpt2_model, 
                                                output_hidden_states=output_hidden_states, 
                                                output_attentions=output_attentions,
                                                pad_token_id=self.tokenizer.eos_token_id)
        self.language = language
        self.pretrained_gpt2_model = pretrained_gpt2_model
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.config = utils.read_yaml(config_path) if config_path else {'max_length': max_length, 
                                                                        'context_length': context_length,
                                                                        'number_of_sentence': number_of_sentence,
                                                                        'number_sentence_before': number_sentence_before}
        self.prediction_type = prediction_type # ['sentence', 'sequential']

    def __name__(self):
        """ Retrieve GPT2 instance name."""
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
            activations = self.get_classic_activations(iterator, language)
            hidden_states_activations = activations[0] 
            attention_heads_activations = activations[1] 
        return [hidden_states_activations, 
                attention_heads_activations]

    def get_classic_activations(self, iterator, language):
        """ Model predictions are generated in the classical way: the model
        take the whole sentence as input.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        # Here, we give as input the batch of line by batch of line.
        batches, indexes = utils.batchify_per_sentence_with_context(iterator, self.config['number_of_sentence'], self.config['number_sentence_before'], self.pretrained_gpt2_model, max_length=self.config['max_length'])
        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=True)
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            attention_mask = torch.tensor([[1 for x in tokenized_text]])
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)

            with torch.no_grad():
                encoded_layers = self.model(inputs_ids, attention_mask=attention_mask) # last_hidden_state, pooler_output, hidden_states, attentions
                # last_hidden_state dimension: (batch_size, sequence_length, hidden_size)
                # pooler_output dimension: (batch_size, hidden_size)
                # hidden_states dimension: num_layers * (batch_size, sequence_length, hidden_size)
                # attentions dimension: num_layers * (batch_size, num_heads, sequence_length, sequence_length)
                # hacked version: attentions dimension: num_layers * [(batch_size, sequence_length, hidden_size), 
                #                                                       (batch_size, num_heads, sequence_length, sequence_length)]
                # filtration
                if self.model.config.output_hidden_states:
                    hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes[index])
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([array[0].view([
                                                                1, 
                                                                inputs_ids.shape[-1], 
                                                                self.NUM_ATTENTION_HEADS, 
                                                                self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS]).permute(0, 2, 1, 3).contiguous()  for array in encoded_layers[3]])
                    attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes[index])
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations]
    