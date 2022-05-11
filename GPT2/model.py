"""
General Language Model based on GPT2 architecture.
The model can implement any pre-trained GPT2 architecture.
This class implements methods to retrieve hidden state or/and
attention heads activations.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from spacy.symbols import ORTH
from transformers import GPT2Tokenizer, GPT2Config

import gpt2_utils
from modeling_hacked_gpt2 import GPT2Model, GPT2LMHeadModel
syntax = __import__('04_syntax_generator')


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
        stop_attention_at_sent_before=None,
        stop_attention_before_sent=0,
        add_prefix_space=True,
        tokens_vocabulary=None,
        pos_dictionary=None,
        prediction=False,
        randomize=False,
        pretrained_gpt2_tokenizer=None
        ):
        super(GPT2Extractor, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_gpt2_tokenizer) if isinstance(pretrained_gpt2_tokenizer, str) else (pretrained_gpt2_tokenizer if pretrained_gpt2_tokenizer is not None else GPT2Tokenizer.from_pretrained(pretrained_gpt2_model))
        gpt2_utils.set_seed(seed)
        if randomize:
            parameters = gpt2_utils.read_yaml(config_path)
            parameters['layer_norm_epsilon'] = float(parameters['layer_norm_epsilon'])
            try:
                self.model = GPT2Model(GPT2Config(**parameters, pad_token_id=self.tokenizer.eos_token_id))
            except:
                self.model = GPT2Model(GPT2Config(**parameters, pad_token_id=1))
            
        elif prediction:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_gpt2_model, 
                                                    output_hidden_states=output_hidden_states, 
                                                    output_attentions=output_attentions,
                                                    pad_token_id=self.tokenizer.eos_token_id)
        else:
            try:
                self.model = GPT2Model.from_pretrained(pretrained_gpt2_model, 
                                                        output_hidden_states=output_hidden_states, 
                                                        output_attentions=output_attentions,
                                                        pad_token_id=self.tokenizer.eos_token_id)
            except:
                self.model = GPT2Model.from_pretrained(pretrained_gpt2_model, 
                                                    output_hidden_states=output_hidden_states, 
                                                    output_attentions=output_attentions,
                                                    pad_token_id=1)
        self.language = language
        self.attention_length_before = attention_length_before
        self.pretrained_gpt2_model = pretrained_gpt2_model
        self.NUM_HIDDEN_LAYERS = self.model.config.num_hidden_layers
        self.FEATURE_COUNT = self.model.config.hidden_size
        self.NUM_ATTENTION_HEADS = self.model.config.num_attention_heads
        self.name = name
        self.add_prefix_space = add_prefix_space
        self.config = {'max_length': max_length, 
                        'context_length': context_length,
                        'number_of_sentence': number_of_sentence,
                        'number_of_sentence_before': number_of_sentence_before,
                        'attention_length_before': attention_length_before,
                        'stop_attention_at_sent_before': stop_attention_at_sent_before, 
                        'stop_attention_before_sent': stop_attention_before_sent,
                        'tokens_vocabulary': tokens_vocabulary,
                        'pos_dictionary': pos_dictionary,
                        'seed': seed,
                        }
        if (config_path is not None) and (not os.path.isdir(config_path)):
            with open(config_path, 'r') as f:  
                self.config.update(json.load(f))

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
        gpt2_utils.set_seed()
        self.model.eval()
        if self.prediction_type in ['sentence', 'sentence-level']:
            activations = self.get_classic_activations(iterator, language)

        elif 'control-context' in self.prediction_type:
            activations = self.get_token_level_activations(iterator, language)
        
        elif 'truncated' in self.prediction_type:
            activations = self.get_truncated_activations(iterator, language)
        
        elif 'shuffle' in self.prediction_type:
            activations = self.get_special_activations(iterator, language, transformation='shuffle')

        elif 'pos' in self.prediction_type:
            activations = self.get_pos_activations(iterator, language)

        hidden_states_activations = activations[0] 
        attention_heads_activations = activations[1] 

        return [hidden_states_activations, 
                attention_heads_activations]
    
    
    def get_truncated_activations(self, iterator, language, bsz=32, space='Ġ', special_token_beg='<|endoftext|>', special_token_end='<|endoftext|>'):
        """Extract hidden state activations of the model for each token from the input, based on truncated input.
        Arguments: 
            - iterator: iterator object
            - language: str
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        """
        hidden_states_activations = []
        attention_heads_activations = []
        iterator = [item.strip() for item in iterator]
        iterator = ' '.join(iterator)
        
        try:
            tokenized_text = self.tokenizer.tokenize(iterator, add_prefix_space=False)
            mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, iterator)
        except:
            print('Failed to used default tokenizer...')
            tokenized_text = self.tokenizer.encode(iterator).tokens
            mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, iterator, eos_token='</s>')
                
        print(f"Using context length of {self.config['context_length']}.")
        
        input_ids, indexes, tokens = gpt2_utils.batchify_to_truncated_input(iterator, self.tokenizer, context_size=self.config['context_length'], max_seq_length=self.config['max_length'], space=space, special_token_beg=special_token_beg, special_token_end=special_token_end)
    
        with torch.no_grad():
            hidden_states_activations_ = []
            for input_tmp in tqdm(input_ids.chunk(input_ids.size(0)//bsz)):
                hidden_states_activations_tmp = []
                encoded_layers = self.model(input_tmp, output_hidden_states=True)
                hidden_states_activations_tmp = np.stack([i.detach().numpy() for i in encoded_layers.hidden_states], axis=0) #shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
                hidden_states_activations_.append(hidden_states_activations_tmp)
                
            hidden_states_activations_ = np.swapaxes(np.vstack([np.swapaxes(item, 0, 1) for item in hidden_states_activations_]), 0, 1) #shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)
            
        activations = []
        for i in range(hidden_states_activations_.shape[1]):
            index = indexes[i]
            activations.append([hidden_states_activations_[:, i, j, :] for j in range(index[0], index[1])])
        activations = np.stack([i for l in activations for i in l], axis=0)
        activations = np.swapaxes(activations, 0, 1) #shape: (#nb_layers, batch_size, hidden_state_dimension)

        for word_index in range(len(mapping.keys())):
            word_activation = []
            word_activation.append([activations[:, index, :] for index in mapping[word_index]])
            word_activation = np.vstack(word_activation)
            hidden_states_activations.append(np.mean(word_activation, axis=0).reshape(-1))# list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
        #After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)
            
        hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        
        return [hidden_states_activations, 
                attention_heads_activations]
    
    def get_pos_activations(self, iterator, language, bsz=16, space='Ġ', special_token_beg=0, special_token_end=2):
        """Extract hidden state activations of the model for each token from the input, based on truncated input.
        Arguments: 
            - iterator: iterator object
            - language: str
        Returns:
            - result: pd.DataFrame containing activation (+ optionally entropy
            and surprisal)
        """
        hidden_states_activations = []
        attention_heads_activations = []
        nlp = syntax.set_nlp_pipeline(name="en_core_web_lg", to_remove=['ner'], max_length=np.inf)
        tokenizer_dict = {
            '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4, 'PUNCT': 5, 'ADV': 6, 
            'AUX': 7, 'SYM': 8, 'ADP': 9, 'SCONJ': 10, 'VERB': 11, 'X': 12, 'PART': 13, 
            'DET': 14, 'NUM': 15, 'NOUN': 16, 'PRON': 17, 'ADJ': 18, 'CCONJ': 19, 'PROPN': 20, 
            'INTJ': 21, 'SPACE': 22
        }
        special_case = [{ORTH: "hasnt"}]
        nlp.tokenizer.add_special_case("hasnt", special_case)
        iterator = [item.strip() for item in iterator]
        iterator = ' '.join(iterator)
        doc = nlp(iterator)
        iterator = []
        tokenized_text = []
        for token in tqdm(doc):
            tokenized_text.append(token.pos_)
            iterator.append(tokenizer_dict[token.pos_])
        mapping = {i:[i] for i, j in enumerate(iterator)}
        
        print(f"Using context length of {self.config['context_length']}.")
        
        input_ids, indexes = gpt2_utils.batchify_pos_input(iterator, context_size=self.config['context_length'], max_seq_length=self.config['max_length'])
    
        with torch.no_grad():
            hidden_states_activations_ = []
            for input_tmp in tqdm(input_ids.chunk(input_ids.size(0)//bsz)):
                hidden_states_activations_tmp = []
                encoded_layers = self.model(input_tmp, output_hidden_states=True)
                hidden_states_activations_tmp = np.stack([i.detach().numpy() for i in encoded_layers.hidden_states], axis=0) #shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
                hidden_states_activations_.append(hidden_states_activations_tmp)
                
            hidden_states_activations_ = np.swapaxes(np.vstack([np.swapaxes(item, 0, 1) for item in hidden_states_activations_]), 0, 1) #shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)
            
        activations = []
        for i in range(hidden_states_activations_.shape[1]):
            index = indexes[i]
            activations.append([hidden_states_activations_[:, i, j, :] for j in range(index[0], index[1])])
        activations = np.stack([i for l in activations for i in l], axis=0)
        activations = np.swapaxes(activations, 0, 1) #shape: (#nb_layers, batch_size, hidden_state_dimension)

        for word_index in range(len(mapping.keys())):
            word_activation = []
            word_activation.append([activations[:, index, :] for index in mapping[word_index]])
            word_activation = np.vstack(word_activation)
            hidden_states_activations.append(np.mean(word_activation, axis=0).reshape(-1))# list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
        #After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)
            
        hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        
        return [hidden_states_activations, 
                attention_heads_activations]

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
        batches, indexes = gpt2_utils.batchify_with_detailed_indexes(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.tokenizer, 
            max_length=self.config['max_length'],
            add_prefix_space=self.add_prefix_space
            )
        indexes_tmp = [(indexes[i][-self.config['number_of_sentence']][0], indexes[i][-1][1]) for i in range(len(indexes))]
        indexes_tmp[0] = (indexes[0][0][0], indexes[0][-1][1])
        
        for i in range(len(indexes_tmp)):
            indexes_tmp[i] = (indexes_tmp[i][0] + 1, indexes_tmp[i][1] + 1)

        for index, batch in enumerate(batches):
            batch = batch.strip() # Remove trailing character
            
            try:
                batch = '<|endoftext|> ' + batch + ' <|endoftext|>' # /!\ depend of the tokenizer used /!\
                tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
                mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, batch)
                inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
            except:
                batch = batch.replace('<|endoftext|> ', '').replace(' <|endoftext|>', '')
                batch = '<s> ' + batch + ' </s>' # /!\ depend of the tokenizer used /!\
                tokenized_text = self.tokenizer.encode(batch).tokens
                mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, batch, eos_token='</s>')
                inputs_ids = torch.tensor([self.tokenizer.encode(batch).ids])
            
            if self.prediction_type == 'sentence':
                attention_mask = torch.tensor([[1 for x in tokenized_text]])
                
            elif self.prediction_type=='sentence-level':
                attention_mask = torch.tensor([[1 for x in tokenized_text]])
                attention_mask[0, : 1+indexes[index][-self.config['stop_attention_at_sent_before']:][0][0]] = 0 # +1 because of special token
              
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
                    hidden_states_activations += gpt2_utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes_tmp[index])
                if self.model.config.output_attentions:
                    attention_heads_activations_ = np.vstack([array[0]  for array in encoded_layers[3]])
                    attention_heads_activations += gpt2_utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index])
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
        batches, indexes = gpt2_utils.batchify_with_detailed_indexes(
            iterator, 
            self.config['number_of_sentence'], 
            self.config['number_of_sentence_before'], 
            self.tokenizer, 
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
            #batch = '<|endoftext|> ' + batch + ' <|endoftext|>'
            batch = '<s> ' + batch + ' </s>'

            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
            mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, batch)
            
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
                    hidden_states_activations += gpt2_utils.extract_activations_from_token_activations(hidden_states_activations_, mapping, indexes_tmp[index_batch])

                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')
                    #attention_heads_activations_ = np.vstack([torch.cat([encoded_layers[-1][layer][0][i,:,i,:].unsqueeze(0) for i in range(len(tokenized_text))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[-1]))])
                    #attention_heads_activations_ = np.swapaxes(attention_heads_activations_, 1, 2)
                    #attention_heads_activations += gpt2_utils.extract_heads_activations_from_token_activations(attention_heads_activations_, mapping, indexes_tmp[index_batch])
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
        batches, indexes = gpt2_utils.batchify_sentences(
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

            #batch = '<|endoftext|> ' + batch + ' <|endoftext|>'
            batch = '<s> ' + batch + ' </s>'
            
            tokenized_text = self.tokenizer.tokenize(batch, add_prefix_space=False)
            #print('Batch number: ', index_batch, ' - ' , batch)
            #print(tokenized_text)
            #print('indexes:', indexes[index_batch], tokenized_text[indexes[index_batch][0]:indexes[index_batch][1]])
            #print()
            inputs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])

            mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, batch)

            with torch.no_grad():
                encoded_layers = self.model(inputs_ids) # last_hidden_state, pooler_output, hidden_states, attentions

                if self.model.config.output_hidden_states:
                    hidden_states_activations_ = np.vstack(encoded_layers[2]) # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                    hidden_states_activations += gpt2_utils.extract_activations_from_token_activations_special(hidden_states_activations_, mapping, indexes[index_batch]) #verify if we have to add 1 to indexes values

                if self.model.config.output_attentions:
                    raise NotImplementedError('Not yet implemented...')

        if self.model.config.output_hidden_states:
            hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + self.NUM_HIDDEN_LAYERS) for index in range(1, 1 + self.FEATURE_COUNT)])
        
        if self.model.config.output_attentions:
            raise NotImplementedError('Not yet implemented...')
        
        return [hidden_states_activations, 
                attention_heads_activations]