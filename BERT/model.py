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
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, BertConfig

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
        attention_length_before=1,
        attention_length_after=1,
        config_path=None, 
        max_length=512, 
        number_of_sentence=1, 
        number_of_sentence_before=0, 
        number_of_sentence_after=0,
        seed=1111,
        hidden_dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        stop_attention_at_sent_before=None,
        stop_attention_before_sent=0,
        tokens_vocabulary=None,
        pos_dictionary=None,
        ):
        super(BertExtractor, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        # Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
        if config_path is None:
            configuration = BertConfig()
            configuration.hidden_dropout_prob = hidden_dropout_prob
            configuration.attention_probs_dropout_prob = attention_probs_dropout_prob
            configuration.output_hidden_states = output_hidden_states
            configuration.output_attentions = output_attentions
            self.model = BertModel.from_pretrained(pretrained_bert_model, config=configuration) #, config=configuration
        else:
            self.model = BertModel.from_pretrained(pretrained_bert_model) #, config=configuration
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_model)
        
        self.language = language
        self.attention_length_before = attention_length_before
        self.attention_length_after = attention_length_after
        self.pretrained_bert_model = pretrained_bert_model
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.config = {'max_length': max_length, 
                        'seed': seed,
                        'number_of_sentence': number_of_sentence,
                        'number_of_sentence_before': number_of_sentence_before,
                        'number_of_sentence_after': number_of_sentence_after,
                        'attention_length_before': attention_length_before,
                        'attention_length_after': attention_length_after,
                        'stop_attention_at_sent_before': stop_attention_at_sent_before, 
                        'stop_attention_before_sent': stop_attention_before_sent,
                        'output_hidden_states': output_hidden_states,
                        'output_attentions': output_attentions,
                        'model_type': 'bert',
                        'tokens_vocabulary': tokens_vocabulary,
                        'pos_dictionary': pos_dictionary,
                        'hidden_size': self.model.config.hidden_size,
                        'hidden_act': self.model.config.hidden_act,
                        'initializer_range': self.model.config.initializer_range,
                        'vocab_size': self.model.config.vocab_size,
                        'hidden_dropout_prob': self.model.config.hidden_dropout_prob,
                        'num_attention_heads': self.model.config.num_attention_heads,
                        'type_vocab_size': self.model.config.type_vocab_size,
                        'max_position_embeddings': self.model.config.max_position_embeddings,
                        'num_hidden_layers': self.model.config.num_hidden_layers,
                        'intermediate_size': self.model.config.intermediate_size,
                        'attention_probs_dropout_prob': self.model.config.attention_probs_dropout_prob
                                                                       }
        if config_path is not None:
            with open(config_path, 'r') as f:  
                self.config.update(json.load(f))

        self.prediction_type = prediction_type # ['sentence', 'token-level']

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
        utils.set_seed(self.config['seed'])
        self.model.eval()
        if self.prediction_type in ['sentence', 'token-level']:
            activations = self.get_classic_activations(iterator, language)
            
        elif 'control-context' in self.prediction_type:
            activations = self.get_token_level_activations(iterator, language)
        
        elif 'truncated' in self.prediction_type:
            activations = self.get_token_level_activations(iterator, language)
        
        elif 'shuffle' in self.prediction_type:
            activations = self.get_special_activations(iterator, language, transformation='shuffle')

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
    
    def get_truncated_activations(self, iterator, language):
        """ Extract hidden state activations of the model for each token from the input, based on truncated input.
        Arguments: 
            - iterator: iterator object
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        """
        hidden_states_activations = []
        attention_heads_activations = []
        cls_hidden_states_activations = []
        sep_hidden_states_activations = []
        cls_attention_activations = []
        sep_attention_activations = []
        # Here, we give as input the batch of line by batch of line.
        batches, mappings = utils.batchify_text_with_memory_size(iterator, self.tokenizer, self.attention_length_before, bos='[CLS]', eos='[SEP]')

        for index, batch in enumerate(batches):

            inputs_ids = torch.tensor(batch)
   
            with torch.no_grad():
                encoded_layers = self.model(inputs_ids) # last_hidden_state, pooler_output, hidden_states, attentions
                # last_hidden_state dimension: (batch_size, sequence_length, hidden_size)
                # pooler_output dimension: (batch_size, hidden_size)
                # hidden_states dimension: num_layers * (batch_size, sequence_length, hidden_size)
                # attentions dimension: num_layers * (batch_size, num_heads, sequence_length, sequence_length)
                # hacked version: attentions dimension: num_layers * [(batch_size, sequence_length, hidden_size), 
                #                                                       (batch_size, num_heads, sequence_length, sequence_length)]
                # filtration
                if self.model.config.output_hidden_states:
                    hidden_states_activations_ = np.stack(encoded_layers[2], axis=0) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations_tmp = []
                    for mapping_index, mapping in enumerate(mappings[index]):
                        if index==0:
                            word_indexes = list(mapping.keys())[1:-1] 
                        else:
                            word_indexes = [list(mapping.keys())[-2]] 
                        for word_index in word_indexes:
                            hidden_states_activations_tmp.append(np.mean(np.array([hidden_states_activations_[:, mapping_index, i, :] for i in mapping[word_index]]), axis=0).reshape(1, -1))
                    hidden_states_activations += hidden_states_activations_tmp
                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
        return [hidden_states_activations, 
                attention_heads_activations, 
                cls_hidden_states_activations,
                sep_hidden_states_activations,
                cls_attention_activations,
                sep_attention_activations]
    
    def get_classic_activations(self, iterator, language):
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
            max_length=self.config['max_length'],
            stop_attention_before_sent=self.config['stop_attention_before_sent'],
            stop_attention_at_sent_before=self.config['stop_attention_at_sent_before']
        )
        
        indexes_tmp = []
        for i in range(len(indexes)):
            if type(indexes[i])==list and type(indexes[i][0])==list:
                indexes_tmp.append(indexes[i][-1])
            else:
                if i > 0:
                    indexes_tmp.append((
                    indexes[i][-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0], 
                    indexes[i][-self.config['number_of_sentence']-self.config['number_of_sentence_after']][1]))
                else:
                    indexes_tmp.append(None)
        
        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '[CLS] ' + batch + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(batch)
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)

            if self.prediction_type=='sentence':
                attention_mask = torch.tensor([[1 for x in tokenized_text]])

                if (self.config['stop_attention_at_sent_before'] is not None) and (index > 0) and not (type(indexes[index])==list and type(indexes[index][0])==list):
                    start_index = 1 if (index > self.config['number_of_sentence_before'] - self.config['stop_attention_at_sent_before'] - self.config['number_of_sentence']) else 0
                    attention_mask[:, :start_index + indexes[index][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]] = 0
                    if self.config['stop_attention_before_sent'] < 0:
                        attention_mask[:, start_index + indexes[index][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]: 1 + indexes[index][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]-self.config['stop_attention_before_sent']] = 0
                    elif self.config['stop_attention_before_sent'] > 0:
                        attention_mask[:, start_index + indexes[index][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]-self.config['stop_attention_before_sent']: 1 + indexes[index][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]] = 1
                elif (self.config['stop_attention_at_sent_before'] is not None) and index > 0:
                    start_index = 1 if (index > self.config['number_of_sentence_before'] - self.config['stop_attention_at_sent_before'] - self.config['number_of_sentence']) else 0
                    attention_mask[:, :start_index + indexes[index][0][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]] = 0
                    if self.config['stop_attention_before_sent'] < 0:
                        attention_mask[:, start_index + indexes[index][0][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]: 1 + indexes[index][0][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]-self.config['stop_attention_before_sent']] = 0
                    elif self.config['stop_attention_before_sent'] > 0:
                        attention_mask[:, start_index + indexes[index][0][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]-self.config['stop_attention_before_sent']: 1 + indexes[index][0][-self.config['stop_attention_at_sent_before']-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0]] = 1

            elif self.prediction_type=='token-level':
                attention_mask =  torch.diag_embed(torch.tensor([0 for x in tokenized_text]))
                for i in range(min(len(tokenized_text), self.attention_length_before)):
                    attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(len(tokenized_text) - i)]), offset=-i))
                for i in range(1, min(len(tokenized_text), self.attention_length_after + 1)):
                    attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(len(tokenized_text) - i)]), offset=i))

                attention_mask = attention_mask.unsqueeze(0)
            
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
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes_tmp[index])
                    #cls_activations_, sep_activations_ = utils.extract_activations_from_special_tokens(hidden_states_activations_, mapping)
                    #cls_hidden_states_activations += cls_activations_
                    #sep_hidden_states_activations += sep_activations_
                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')
                    #attention_heads_activations_ = np.vstack([array[0].view([
                    #                                            1, 
                    #                                            inputs_ids.shape[-1], 
                    #                                            self.NUM_ATTENTION_HEADS, 
                    #                                            self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS]).permute(0, 2, 1, 3).contiguous()  for array in encoded_layers[3]])
                    #attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index])
                    ##cls_attention_, sep_attention_ = utils.extract_heads_activations_from_special_tokens(attention_heads_activations_, mapping)
                    ##cls_attention_activations += cls_attention_
                    ##sep_attention_activations += sep_attention_
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            #cls_hidden_states_activations = pd.DataFrame(np.vstack(cls_hidden_states_activations), columns=['CLS-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            #sep_hidden_states_activations = pd.DataFrame(np.vstack(sep_hidden_states_activations), columns=['SEP-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
            #attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            ##cls_attention_activations = pd.DataFrame(np.vstack(cls_attention_activations), columns=['CLS-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            ##sep_attention_activations = pd.DataFrame(np.vstack(sep_attention_activations), columns=['SEP-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
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
        
        indexes_tmp = []
        # If beginning and end indexes of each sentences are recorded, we only keep the sentence(s) of interest
        for i in range(len(indexes)):
            if type(indexes[i])==list and type(indexes[i][0])==list:
                indexes_tmp.append(indexes[i][-1])
            else:
                if i > 0:
                    indexes_tmp.append((
                    indexes[i][-self.config['number_of_sentence']-self.config['number_of_sentence_after']][0], 
                    indexes[i][-self.config['number_of_sentence']-self.config['number_of_sentence_after']][1]))
                else:
                    indexes_tmp.append(None)
        
        for index_batch, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '[CLS] ' + batch + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(batch)
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            inputs_ids = torch.cat(inputs_ids.size(1) * [inputs_ids])
            attention_mask =  torch.diag_embed(torch.tensor([[0 for x in tokenized_text]]))

            for i in range(min(len(tokenized_text), self.attention_length_before)):
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([[1 for x in range(len(tokenized_text) - i)]]), offset=-i))
            for i in range(1, min(len(tokenized_text), self.attention_length_after + 1)):
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([[1 for x in range(len(tokenized_text) - i)]]), offset=i))
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)

            attention_mask = attention_mask.squeeze(0)

            beg = indexes_tmp[index_batch][0] + 1 # because of the special token at the beginning
            end = indexes_tmp[index_batch][1] + 1 # because of special token

            inputs_ids = inputs_ids[beg:end, :]
            attention_mask = attention_mask[beg:end, :]

            dim = inputs_ids.size(1)
            if self.prediction_type=='control-context-past':
                attention_mask = torch.stack([attention_mask[index, :] * torch.tril(torch.ones(dim, dim)) for index in range(attention_mask.size(0))])
            elif self.prediction_type=='control-context-future':
                attention_mask = torch.stack([attention_mask[index, :] * torch.triu(torch.ones(dim, dim)) for index in range(attention_mask.size(0))])
                            
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
                    hidden_states_activations_ = np.vstack([torch.cat([encoded_layers[2][layer][i,len(tokenized_text) - encoded_layers[2][layer].size(0) + i - 1,:].unsqueeze(0) for i in range(encoded_layers[2][layer].size(0))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[2]))]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations_ = np.concatenate([np.zeros((hidden_states_activations_.shape[0], indexes_tmp[index_batch][0] + 1 , hidden_states_activations_.shape[-1])), hidden_states_activations_, np.zeros((hidden_states_activations_.shape[0], len(tokenized_text) - indexes_tmp[index_batch][1] - 1, hidden_states_activations_.shape[-1]))], axis=1)

                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes_tmp[index_batch])
                    #cls_activations_, sep_activations_ = utils.extract_activations_from_special_tokens(hidden_states_activations_, mapping)
                    #cls_hidden_states_activations += cls_activations_
                    #sep_hidden_states_activations += sep_activations_
                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')
                    #attention_heads_activations_ = np.vstack([torch.cat([encoded_layers[-1][layer][0][i,len(tokenized_text) - encoded_layers[-1][layer][0].size(0) + i,:].unsqueeze(0) for i in range(encoded_layers[-1][layer][0].size(0))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[-1]))])
                    #if indexes_tmp[index_batch][0] > 0:
                    #    attention_heads_activations_ = np.concatenate([np.zeros((attention_heads_activations_.shape[0], indexes_tmp[index_batch][0] , attention_heads_activations_.shape[-1])), attention_heads_activations_], axis=1)
                    #attention_heads_activations_ = attention_heads_activations_.reshape([
                    #    self.NUM_HIDDEN_LAYERS, 
                    #    len(tokenized_text), 
                    #    self.NUM_ATTENTION_HEADS, 
                    #    self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS])
                    #attention_heads_activations_ = np.swapaxes(attention_heads_activations_, 1, 2)
                    #attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index_batch])
                    #cls_attention_, sep_attention_ = utils.extract_heads_activations_from_special_tokens(attention_heads_activations_, mapping)
                    #cls_attention_activations += cls_attention_
                    #sep_attention_activations += sep_attention_
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            #cls_hidden_states_activations = pd.DataFrame(np.vstack(cls_hidden_states_activations), columns=['CLS-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
            #sep_hidden_states_activations = pd.DataFrame(np.vstack(sep_hidden_states_activations), columns=['SEP-hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
            #attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            #cls_attention_activations = pd.DataFrame(np.vstack(cls_attention_activations), columns=['CLS-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
            #sep_attention_activations = pd.DataFrame(np.vstack(sep_attention_activations), columns=['SEP-attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations, 
                cls_hidden_states_activations,
                sep_hidden_states_activations,
                cls_attention_activations,
                sep_attention_activations]
    
    def get_special_activations(self, iterator, language, transformation):
        """ Model predictions are generated by batch and apply different transformation to the input: the model
        take the whole sentence as input.
        For each word, words outside its context window are either:
            - shuffled
            - replaced by random words
            - replaced by words with same POS and relationship dependencies
        """
        hidden_states_activations = []
        attention_heads_activations = []
        cls_hidden_states_activations = []
        sep_hidden_states_activations = []
        cls_attention_activations = []
        sep_attention_activations = []
        # Here, a batch is juste a sentence because we cannot create batches of equal length due to the transformation
        batches, indexes = utils.batchify_sentences(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.config['number_of_sentence_after'], 
            self.pretrained_bert_model, 
            past_context_size=self.config['attention_length_before'],
            future_context_size=self.config['attention_length_after'],
            transformation=transformation,
            vocabulary=self.config['tokens_vocabulary'],
            dictionary=self.config['pos_dictionary'],
            seed=self.config['seed'],
            max_length=self.config['max_length'])
        
        for index_batch, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '[CLS] ' + batch + ' [SEP]'
            tokenized_text = self.tokenizer.wordpiece_tokenizer.tokenize(batch)
            #print('Batch number: ', index_batch, ' - ' , batch)
            #print(tokenized_text)
            #print('indexes:', indexes[index_batch], tokenized_text[indexes[index_batch][0]:indexes[index_batch][1]])
            #print()
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])

            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)

            with torch.no_grad():
                encoded_layers = self.model(inputs_ids) # last_hidden_state, pooler_output, hidden_states, attentions

                if self.model.config.output_hidden_states:
                    hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes[index_batch]) #verify if we have to add 1 to indexes values

                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')

        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
        
        return [hidden_states_activations, 
                attention_heads_activations, 
                cls_hidden_states_activations,
                sep_hidden_states_activations,
                cls_attention_activations,
                sep_attention_activations]