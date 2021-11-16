import os
import glob
import torch
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import GPT2Extractor
from sklearn.preprocessing import StandardScaler
from tokenizer import tokenize
from gpt2_utils import set_seed
from numpy import linalg as la



#### Functions ####

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def transform(activations, path, name, run_index, n_layers_hidden=13, n_layers_attention=12, hidden_size=768):
    assert activations.values.shape[1] == (n_layers_hidden + n_layers_attention) * hidden_size
    indexes = [[index*hidden_size, (index+1)*hidden_size] for index in range(n_layers_hidden + n_layers_attention)]
    for order in [2]:
        matrices = []
        for index in indexes:
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


#### Variables ####
template = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/text_english_run*.txt' # path to text input
language = 'english'
saving_path_folder = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/stimuli-representations/{}'.format(language)

#### Preprocessing ####
paths = sorted(glob.glob(template))
iterator_list = [tokenize(path, language, train=False) for path in paths]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract GPT2 activations')
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--prediction_type", type=str)
    parser.add_argument("--number_of_sentence", type=int)
    parser.add_argument("--number_of_sentence_before", type=int)
    parser.add_argument("--attention_length_before", type=int)
    parser.add_argument("--stop_attention_at_sent_before", type=int)

    args = parser.parse_args()


    pretrained_gpt2_models = [args.model]
    names = [args.name]
    seed = args.seed
    randomize = args.randomize
    config_paths = [args.config_path]
    saving_path_folders = [os.path.join(saving_path_folder, args.name)]
    prediction_types = [args.prediction_type]
    number_of_sentence_list = [args.number_of_sentence]
    number_of_sentence_before_list = [args.number_of_sentence_before] 
    attention_length_before_list = [args.attention_length_before]
    stop_attention_at_sent_before_list = [args.stop_attention_at_sent_before]

    output_attentions = False
    output_hidden_states = True

    #### Computations ####

    for index, gpt2_model in enumerate(pretrained_gpt2_models):
        extractor = GPT2Extractor(gpt2_model, 
                                language, 
                                names[index], 
                                prediction_types[index],
                                output_hidden_states=output_hidden_states, 
                                output_attentions=output_attentions,
                                attention_length_before=attention_length_before_list[index],
                                config_path=config_paths[index],
                                max_length=512, 
                                seed=seed,
                                randomize=randomize,
                                number_of_sentence=number_of_sentence_list[index], 
                                number_of_sentence_before=number_of_sentence_before_list[index],
                                stop_attention_at_sent_before=stop_attention_at_sent_before_list[index],
                                )
        print(extractor.name, ' - Extracting activations ...')
        for run_index, iterator in tqdm(enumerate(iterator_list)):
            gc.collect()
            print("############# Run {} #############".format(run_index + 1))
            activations  = extractor.extract_activations(iterator, language)
            hidden_states_activations = activations[0]
            
            #check_folder(saving_path_folders[index])
            #hidden_states_activations.to_csv(os.path.join(saving_path_folders[index], 'activations_run{}.csv'.format(run_index + 1)), index=False)
            transform(
                hidden_states_activations, 
                saving_path_folders[index], 
                'activations', 
                run_index=run_index,
                n_layers_hidden=extractor.config['n_layer'] + 1 if output_hidden_states else 0,
                n_layers_attention=extractor.config['n_layer'] if output_attentions else 0, 
                hidden_size=extractor.config['n_embd'])

            del activations
            del hidden_states_activations
