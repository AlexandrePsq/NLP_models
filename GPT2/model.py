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

    def __init__(
        self, 
        pretrained_gpt2_model, 
        language, 
        name, 
        prediction_type, 
        output_hidden_states, 
        output_attentions, 
        attention_length_before=1,
        config_path=None, 
        max_length=512, 
        seed=1111,
        context_length=250, 
        number_of_sentence=1, 
        number_of_sentence_before=0,
        stop_attention_at_sent=None,
        stop_attention_before_sent=0,
        add_prefix_space=True,
        tokens_vocabulary=None,
        pos_dictionary=None,
        ):
        super(GPT2Extractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_gpt2_model)
        self.model = GPT2Model.from_pretrained(pretrained_gpt2_model, 
                                                output_hidden_states=output_hidden_states, 
                                                output_attentions=output_attentions,
                                                pad_token_id=self.tokenizer.eos_token_id)
        self.language = language
        self.attention_length_before = attention_length_before
        self.pretrained_gpt2_model = pretrained_gpt2_model
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.add_prefix_space = add_prefix_space
        self.config = utils.read_yaml(config_path) if config_path else {'max_length': max_length, 
                                                                        'context_length': context_length,
                                                                        'number_of_sentence': number_of_sentence,
                                                                        'number_of_sentence_before': number_of_sentence_before,
                                                                        'attention_length_before': attention_length_before,
                                                                        'stop_attention_at_sent': stop_attention_at_sent, 
                                                                        'stop_attention_before_sent': stop_attention_before_sent,
                                                                        'tokens_vocabulary': tokens_vocabulary,
                                                                        'pos_dictionary': pos_dictionary,
                                                                        'seed': seed,
                                                                        }
        self.prediction_type = prediction_type # ['sentence', 'token-level']

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
        if self.prediction_type in ['sentence', 'token-level']:
            activations = self.get_classic_activations(iterator, language)

        elif 'control-context' in self.prediction_type:
            activations = self.get_token_level_activations(iterator, language)
        
        elif 'truncated' in self.prediction_type:
            activations = self.get_truncated_activations(iterator, language)
        
        elif 'shuffle' in self.prediction_type:
            activations = self.get_special_activations(iterator, language, transformation='shuffle')

        hidden_states_activations = activations[0] 
        attention_heads_activations = activations[1] 

        return [hidden_states_activations, 
                attention_heads_activations]
    
    def get_truncated_activations(self, iterator, language):
        """ Extract hidden state activations of the model for each token from the input, based on truncated input.
        Arguments: 
            - iterator: iterator object
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        """
        raise NotImplementedError('Not yet implemented...')
        #hidden_states_activations = []
        #attention_heads_activations = []
        ## Here, we give as input the batch of line by batch of line.
        #batches, mappings = utils.batchify_text_with_memory_size(iterator, self.tokenizer, self.attention_length_before, bos='<|endoftext|>', eos='<|endoftext|>')

        #for index, batch in enumerate(batches):

        #    inputs_ids = torch.tensor(batch)

        #    with torch.no_grad():
        #        encoded_layers = self.model(inputs_ids) # last_hidden_state, pooler_output, hidden_states, attentions
        #        # last_hidden_state dimension: (batch_size, sequence_length, hidden_size)
        #        # pooler_output dimension: (batch_size, hidden_size)
        #        # hidden_states dimension: num_layers * (batch_size, sequence_length, hidden_size)
        #        # attentions dimension: num_layers * (batch_size, num_heads, sequence_length, sequence_length)
        #        # hacked version: attentions dimension: num_layers * [(batch_size, sequence_length, hidden_size), 
        #        #                                                       (batch_size, num_heads, sequence_length, sequence_length)]
        #        # filtration
        #        if self.model.config.output_hidden_states:
        #            hidden_states_activations_ = np.stack(encoded_layers[2], axis=0) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
        #            hidden_states_activations_tmp = []
        #            for mapping_index, mapping in enumerate(mappings[index]):
        #                if index==0:
        #                    word_indexes = list(mapping.keys())[1:-2] # -2 because the special token at the end is tokenized into two by the tokenizer
        #                else:
        #                    word_indexes = [list(mapping.keys())[-3]] # same reason
        #                for word_index in word_indexes:
        #                    hidden_states_activations_tmp.append(np.mean(np.array([hidden_states_activations_[:, mapping_index, i, :] for i in mapping[word_index]]), axis=0).reshape(1, -1))
        #            hidden_states_activations += hidden_states_activations_tmp
        #        if self.model.config.output_attentions:
        #            raise NotImplementedError('Not yet implemented...')
        #if self.model.config.output_hidden_states:
        #    hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        #if self.model.config.output_attentions:
        #    raise NotImplementedError('Not yet implemented...')
        #return [hidden_states_activations, 
        #        attention_heads_activations]


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
        # Here, we give as input the batch of line by batch of line.
        batches, indexes = utils.batchify_with_detailed_indexes(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.pretrained_gpt2_model, 
            max_length=self.config['max_length'],
            stop_attention_at_sent=self.config['stop_attention_at_sent'],
            stop_attention_before_sent=self.config['stop_attention_before_sent'],
            add_prefix_space=self.add_prefix_space
            )
        indexes_tmp = [(indexes[i][-self.config['number_of_sentence']][0], indexes[i][-1][1]) for i in range(len(indexes))]
        indexes_tmp[0] = (indexes[0][0][0], indexes[0][-1][1])
        
        for i in range(len(indexes_tmp)):
            indexes_tmp[i] = (indexes_tmp[i][0] + 1, indexes_tmp[i][1] + 1)

        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character
            batch = '<|endoftext|> ' + batch + ' <|endoftext|>'

            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            
            if self.prediction_type == 'sentence':
                attention_mask = torch.tensor([[1 for x in tokenized_text]])

                if (self.config['stop_attention_at_sent'] is not None) and (index > 0):
                    attention_mask[:, : indexes[index][-self.config['stop_attention_at_sent']-self.config['number_of_sentence']][0]] = 0
                    if self.config['stop_attention_before_sent'] < 0:
                        attention_mask[:, 1 + indexes[index][-self.config['stop_attention_at_sent']-self.config['number_of_sentence']][0]: 1 + indexes[index][-self.config['stop_attention_at_sent']-self.config['number_of_sentence']][0]-self.config['stop_attention_before_sent']] = 0
                    elif self.config['stop_attention_before_sent'] > 0:
                        attention_mask[:, 1 + indexes[index][-self.config['stop_attention_at_sent']-self.config['number_of_sentence']][0]-self.config['stop_attention_before_sent']: 1 + indexes[index][-self.config['stop_attention_at_sent']-self.config['number_of_sentence']][0]] = 1
                        
            elif 'token-level' in self.prediction_type:
                attention_mask =  torch.diag_embed(torch.tensor([0 for x in tokenized_text]))
                for i in range(min(len(tokenized_text), self.attention_length_before)):
                    attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(len(tokenized_text) - i)]), offset=-i))
                attention_mask = attention_mask.unsqueeze(0)
                if 'reverse' in self.prediction_type:
                    attention_mask = 1 - attention_mask
                               
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
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([array[0]  for array in encoded_layers[3]])
                    attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index])
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations]
    
    
    def get_token_level_activations(self, iterator, language):
        """ Model predictions are generated by batch with small attention masks: the model
        take the whole sentence as input.
        """
        hidden_states_activations = []
        attention_heads_activations = []
        # Here, we give as input the batch of line by batch of line.
        batches, indexes = utils.batchify_with_detailed_indexes(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.pretrained_gpt2_model, 
            max_length=self.config['max_length'],
            add_prefix_space=self.add_prefix_space
        )
        indexes_tmp = [(indexes[i][-self.config['number_of_sentence']][0], indexes[i][-1][1]) for i in range(len(indexes))]
        indexes_tmp[0] = (indexes[0][0][0], indexes[0][-1][1])
        # we ad 1 because of the initial special token
        for i in range(len(indexes_tmp)):
            indexes_tmp[i] = (indexes_tmp[i][0] + 1, indexes_tmp[i][1] + 1)
            
        for index_batch, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character
            batch = '<|endoftext|> ' + batch + ' <|endoftext|>'

            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
            mapping = utils.match_tokenized_to_untokenized(tokenized_text, batch)
            
            beg = indexes_tmp[index_batch][0] 
            end = indexes_tmp[index_batch][1] 

            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            inputs_ids = torch.cat(inputs_ids.size(1) * [inputs_ids])
            inputs_ids = inputs_ids[beg:end, :]

            attention_mask =  torch.diag_embed(torch.tensor([0 for x in tokenized_text]))
            for i in range(min(len(tokenized_text), self.attention_length_before)):
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(len(tokenized_text) - i)]), offset=-i))
            attention_mask = attention_mask[beg:end, :]

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
                    hidden_states_activations_ = np.vstack([torch.cat([encoded_layers[2][layer][i,len(tokenized_text) - encoded_layers[2][layer].size(0) + i - 1,:].unsqueeze(0) for i in range(encoded_layers[2][layer].size(0))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[2]))])
                    hidden_states_activations_ = np.concatenate([np.zeros((hidden_states_activations_.shape[0], indexes_tmp[index_batch][0] , hidden_states_activations_.shape[-1])), hidden_states_activations_, np.zeros((hidden_states_activations_.shape[0], len(tokenized_text) - indexes_tmp[index_batch][1], hidden_states_activations_.shape[-1]))], axis=1)
                    # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes_tmp[index_batch])

                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')
                    #attention_heads_activations_ = np.vstack([torch.cat([encoded_layers[-1][layer][0][i,:,i,:].unsqueeze(0) for i in range(len(tokenized_text))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[-1]))])
                    #attention_heads_activations_ = np.swapaxes(attention_heads_activations_, 1, 2)
                    #attention_heads_activations += utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index_batch])
        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
            #attention_heads_activations = pd.DataFrame(np.vstack(attention_heads_activations), columns=['attention-layer-{}-head-{}-{}'.format(layer, head, index) for layer in np.arange(1, 1 + self.NUM_HIDDEN_LAYERS) for head in range(1, 1 + self.NUM_ATTENTION_HEADS) for index in range(1, 1 + self.FEATURE_COUNT // self.NUM_ATTENTION_HEADS)])
        return [hidden_states_activations, 
                attention_heads_activations]

    
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
        # Here, a batch is juste a sentence because we cannot create batches of equal length due to the transformation
        batches, indexes = utils.batchify_sentences(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.pretrained_gpt2_model, 
            past_context_size=self.config['attention_length_before'],
            transformation=transformation,
            vocabulary=self.config['tokens_vocabulary'],
            dictionary=self.config['pos_dictionary'],
            seed=self.config['seed'],
            max_length=self.config['max_length'])
        
        for index_batch, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character

            batch = '<|endoftext|> ' + batch + ' <|endoftext|>'
            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
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
                attention_heads_activations]