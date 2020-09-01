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

    def __init__(
        self, 
        pretrained_bert_model, 
        language, 
        name, 
        prediction_type,
        output_hidden_states, 
        output_attentions, 
        attention_length=1,
        config_path=None, 
        max_length=512, 
        context_length=250, 
        number_of_sentence=1, 
        number_of_sentence_before=0, 
        number_of_sentence_after=0
        ):
        super(BertExtractor, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        self.model = BertModel.from_pretrained(pretrained_bert_model, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_model)
        
        self.language = language
        self.attention_length = attention_length
        self.pretrained_bert_model = pretrained_bert_model
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.config = utils.read_yaml(config_path) if config_path else {'max_length': max_length, 
                                                                        'context_length': context_length,
                                                                        'number_of_sentence': number_of_sentence,
                                                                        'number_of_sentence_before': number_of_sentence_before,
                                                                        'number_of_sentence_after': number_of_sentence_after,
                                                                        'attention_length': attention_length}
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
            activations = self.get_classic_activations(iterator, language)
            hidden_states_activations = activations[0] 
            attention_heads_activations = activations[1] 
            cls_hidden_states_activations = activations[2]
            sep_hidden_states_activations = activations[3] 
            cls_attention_activations = activations[4]
            sep_attention_activations = activations[5] 
        elif self.prediction_type == 'token-level':
            activations = self.get_token_level_activations(iterator, language)
            hidden_states_activations = activations[0] 
            attention_heads_activations = activations[1] 
            cls_hidden_states_activations = activations[2]
            sep_hidden_states_activations = activations[3] 
            cls_attention_activations = activations[4]
            sep_attention_activations = activations[5] 
        return [hidden_states_activations, 
                attention_heads_activations, 
                (cls_hidden_states_activations, cls_attention_activations),
                (sep_hidden_states_activations, sep_attention_activations)]

    def get_classic_activations(self, iterator, language):
        """ Model predictions are generated in the classical way: the model
        take the whole sentence as input.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        cls_hidden_states_activations = []
        sep_hidden_states_activations = []
        cls_attention_activations = []
        sep_attention_activations = []
        # Here, we give as input the batch of line by batch of line.
        batches, indexes = utils.batchify_per_sentence_with_pre_and_post_context(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.config['number_of_sentence_after'], 
            self.pretrained_bert_model, 
            max_length=self.config['max_length'])
            
        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '[CLS] ' + batch + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(batch)
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
                    cls_activations_, sep_activations_ = utils.extract_activations_from_special_tokens(hidden_states_activations_, mapping)
                    cls_hidden_states_activations += cls_activations_
                    sep_hidden_states_activations += sep_activations_
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([array[0].view([
                                                                1, 
                                                                inputs_ids.shape[-1], 
                                                                self.NUM_ATTENTION_HEADS, 
                                                                self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS]).permute(0, 2, 1, 3).contiguous()  for array in encoded_layers[3]])
                    attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes[index])
                    cls_attention_, sep_attention_ = utils.extract_heads_activations_from_special_tokens(attention_heads_activations_, mapping)
                    cls_attention_activations += cls_attention_
                    sep_attention_activations += sep_attention_
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            cls_hidden_states_activations = pd.DataFrame(np.vstack(cls_hidden_states_activations), columns=['CLS-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            sep_hidden_states_activations = pd.DataFrame(np.vstack(sep_hidden_states_activations), columns=['SEP-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            cls_attention_activations = pd.DataFrame(np.vstack(cls_attention_activations), columns=['CLS-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            sep_attention_activations = pd.DataFrame(np.vstack(sep_attention_activations), columns=['SEP-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations, 
                cls_hidden_states_activations,
                sep_hidden_states_activations,
                cls_attention_activations,
                sep_attention_activations]
    

    def get_token_level_activations(self, iterator, language):
        """ Model predictions are generated by batch with small attention masks: the model
        take the whole sentence as input.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        cls_hidden_states_activations = []
        sep_hidden_states_activations = []
        cls_attention_activations = []
        sep_attention_activations = []
        # Here, we give as input the batch of line by batch of line.
        batches, indexes = utils.batchify_per_sentence_with_pre_and_post_context(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.config['number_of_sentence_after'], 
            self.pretrained_bert_model, 
            max_length=self.config['max_length'])
            
        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '[CLS] ' + batch + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(batch)
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            inputs_ids = torch.cat(inputs_ids.size(1) * [inputs_ids])
            attention_mask =  torch.diag_embed(torch.tensor([[0 for x in tokenized_text]]))
            for i in range(min(len(tokenized_text), self.attention_length)):
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([[1 for x in range(len(tokenized_text) - i)]]), offset=-i))
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
                    hidden_states_activations_ = np.vstack([torch.cat([encoded_layers[2][layer][i,i,:].unsqueeze(0) for i in range(len(tokenized_text))], axis=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[2]))]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes[index])
                    cls_activations_, sep_activations_ = utils.extract_activations_from_special_tokens(hidden_states_activations_, mapping)
                    cls_hidden_states_activations += cls_activations_
                    sep_hidden_states_activations += sep_activations_
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([torch.cat([encoded_layers[-1][layer][0][i,i,:].unsqueeze(0) for i in range(len(tokenized_text))], axis=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[-1]))])
                    attention_heads_activations_ = attention_heads_activations_.reshape([
                        self.NUM_HIDDEN_LAYERS, 
                        len(tokenized_text), 
                        self.NUM_ATTENTION_HEADS, 
                        self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS])   
                    attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes[index])
                    cls_attention_, sep_attention_ = utils.extract_heads_activations_from_special_tokens(attention_heads_activations_, mapping)
                    cls_attention_activations += cls_attention_
                    sep_attention_activations += sep_attention_
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            cls_hidden_states_activations = pd.DataFrame(np.vstack(cls_hidden_states_activations), columns=['CLS-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            sep_hidden_states_activations = pd.DataFrame(np.vstack(sep_hidden_states_activations), columns=['SEP-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            cls_attention_activations = pd.DataFrame(np.vstack(cls_attention_activations), columns=['CLS-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            sep_attention_activations = pd.DataFrame(np.vstack(sep_attention_activations), columns=['SEP-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations, 
                cls_hidden_states_activations,
                sep_hidden_states_activations,
                cls_attention_activations,
                sep_attention_activations]
    
 