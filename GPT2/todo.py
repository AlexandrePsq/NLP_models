import os
import glob
import torch
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import GPT2Extractor
from sklearn.preprocessing import StandardScaler
from tokenizer import tokenize
#from utils import set_seed
from numpy import linalg as la
import matplotlib.pyplot as plt
import random
from transformers import GPT2Tokenizer
from sklearn.decomposition import PCA
from collections import defaultdict

import utils
import gpt2_utils

def transform(activations, path, name, run_index, n_layers_hidden=13, n_layers_attention=12, hidden_size=768):
    assert activations.values.shape[1] == (n_layers_hidden + n_layers_attention) * hidden_size
    indexes = [[index*hidden_size, (index+1)*hidden_size] for index in range(n_layers_hidden + n_layers_attention)]
    for order in [2]: # np.inf
        matrices = []
        for i, index in enumerate(indexes):
            matrix = activations.values[:, index[0]:index[1]]
            #with_std = True if order=='std' else False
            #scaler = StandardScaler(with_mean=True, with_std=with_std)
            #scaler.fit(matrix)
            #matrix = scaler.transform(matrix)
            if order is not None and order != 'std':
                matrix = matrix / np.mean(la.norm(matrix, ord=order, axis=1))
            matrices.append(matrix)
        matrices = np.hstack(matrices)
        new_data = pd.DataFrame(matrices, columns=activations.columns)
        new_path = path + '_norm-' + str(order).replace('np.', '')
        check_folder(new_path)
        new_data.to_csv(os.path.join(new_path, name + '_run{}.csv'.format(run_index + 1)), index=False)

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent, connection_character='Ġ', eos_token='<|endoftext|>'):
    '''Aligns tokenized and untokenized sentence given non-subwords "Ġ" prefixed
    Assuming that each subword token that does start a new word is prefixed
    by "Ġ", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.
    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 0
    while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
        while (tokenized_sent_index + 1  < len(tokenized_sent) and (not tokenized_sent[tokenized_sent_index + 1].startswith(connection_character)) and tokenized_sent[tokenized_sent_index+1]!=eos_token):
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1
    return mapping


template = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/text_english_run*.txt' # path to text input
language = 'english'


paths = sorted(glob.glob(template))
iterator_list = [tokenize(path, language, train=False) for path in paths]




for attention_length_before in [1000]:

    config = {
        'number_of_sentence': 1, 
        'number_of_sentence_before': 10, 
        'attention_length_before': attention_length_before, 
        'stop_attention_at_sent_before': 1,
    }



    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    extractor_masked = GPT2Extractor('gpt2', 
                                  'english', 
                                  'test', 
                                  'control-context', 
                                  output_hidden_states=True, 
                                  output_attentions=False, 
                                  attention_length_before=config['attention_length_before'],
                                  config_path=None, 
                                  number_of_sentence=config['number_of_sentence'], 
                                  number_of_sentence_before=config['number_of_sentence_before'], 
                                  prediction=True
                                 )
    
    for run in range(9):
        # Tokens are masked
        batches_masked, indexes_masked = gpt2_utils.batchify_with_detailed_indexes(
                    iterator_list[run], 
                    config['number_of_sentence'], 
                    config['number_of_sentence_before'], 
                    'gpt2',
                    add_prefix_space=True
                    )

        # Preprocessing masked
        indexes_masked_tmp = [(indexes_masked[i][-config['number_of_sentence']][0], indexes_masked[i][-1][1]) for i in range(len(indexes_masked))]
        indexes_masked_tmp[0] = (indexes_masked[0][0][0], indexes_masked[0][-1][1])

        # activation generation masked
        output = []
        activations = []
        for index_batch, batch in tqdm(enumerate(batches_masked)):
            batch = batch.strip() # Remove trailing character
            batch = '<|endoftext|> ' + batch + ' <|endoftext|>'

            tokenized_text = tokenizer.tokenize(batch, add_prefix_space=False)
            mapping = gpt2_utils.match_tokenized_to_untokenized(tokenized_text, batch)

            beg = indexes_masked_tmp[index_batch][0] 
            end = indexes_masked_tmp[index_batch][1] 

            inputs_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
            inputs_ids = torch.cat(inputs_ids.size(1) * [inputs_ids])
            inputs_ids = inputs_ids[beg:end, :]

            attention_mask =  torch.diag_embed(torch.tensor([0 for x in tokenized_text]))
            for i in range(min(len(tokenized_text), config['attention_length_before'])):
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(len(tokenized_text) - i)]), offset=-i))
            attention_mask = attention_mask[beg:end, :]

            with torch.no_grad():
                encoded_layers = extractor_masked.model(inputs_ids, attention_mask=attention_mask, labels=inputs_ids, return_dict=False) # last_hidden_state, pooler_output, hidden_states, attentions       
                #print(beg, len(tokenized_text) - encoded_layers[2][0].size(0) - 1)
                #hidden_states_activations_ = np.vstack([torch.cat([encoded_layers[2][layer][i,len(tokenized_text) - encoded_layers[2][layer].size(0) + i - 1,:].unsqueeze(0) for i in range(encoded_layers[2][layer].size(0))], dim=0).unsqueeze(0).detach().numpy() for layer in range(len(encoded_layers[2]))])
                #hidden_states_activations_ = np.concatenate([np.zeros((hidden_states_activations_.shape[0], indexes_masked_tmp[index_batch][0] , hidden_states_activations_.shape[-1])), hidden_states_activations_, np.zeros((hidden_states_activations_.shape[0], len(tokenized_text) - indexes_masked_tmp[index_batch][1], hidden_states_activations_.shape[-1]))], axis=1)
                # retrieve all the hidden states (dimension = layer_count * len(tokenized_text) * feature_count)
                loss_ = torch.cat([encoded_layers[0][i,len(tokenized_text) - encoded_layers[0].size(0) + i - 2].unsqueeze(0) for i in range(encoded_layers[0].size(0))], dim=0).unsqueeze(0).detach().numpy()

                activations.append(loss_)

        np.save(os.path.join(
            '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/16_masking_before_vs_after_softmax', 
            f"masking_after_at_{config['attention_length_before']}_run{run+1}.npy"), 
                np.hstack(activations).reshape(-1)
               )
