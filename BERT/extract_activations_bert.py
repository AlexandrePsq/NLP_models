import os
import glob
import torch
import gc
import spacy
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import BertExtractor
from sklearn.preprocessing import StandardScaler
from tokenizer import tokenize
from numpy import linalg as la

from get_dependency_parsing import generate_pos_freq_sample, generate_freq_sample, filter_list

#### Functions ####


def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
            quit()
    return parameters

def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
def write(path, text, end='\n'):
    """Write in the specified text file."""
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)

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
nlp = spacy.load("en_core_web_lg")
nlp.remove_pipe("ner")
nlp.max_length = np.inf
english_words_data = pd.read_csv('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/lexique_database.tsv', delimiter='\t')
# Creating dict with freq information
word_list = english_words_data['Word'].apply(lambda x: str(x).lower()).values
freq_list = english_words_data['Lg10WF'].values
zip_freq = zip(word_list, freq_list)
word_freq = dict(zip_freq)

#### Preprocessing ####
paths = sorted(glob.glob(template))
iterator_list = [tokenize(path, language, train=False) for path in paths]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract BERT activations')
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--prediction_type", type=str)
    parser.add_argument("--number_of_sentence", type=int)
    parser.add_argument("--number_of_sentence_before", type=int)
    parser.add_argument("--number_of_sentence_after", type=int)
    parser.add_argument("--attention_length_before", type=int)
    parser.add_argument("--attention_length_after", type=int)
    parser.add_argument("--nb_random_sample", type=int, default=0)
    parser.add_argument("--same_freq", type=bool, default=False)
    parser.add_argument("--same_syntax", type=bool, default=False)

    args = parser.parse_args()


    pretrained_bert_models = [args.model]
    names = [args.name]
    config_paths = [args.config_path]
    saving_path_folders = [os.path.join(saving_path_folder, args.name)]
    prediction_types = [args.prediction_type]
    number_of_sentence_list = [args.number_of_sentence]
    number_of_sentence_before_list = [args.number_of_sentence_before] 
    number_of_sentence_after_list = [args.number_of_sentence_after] 
    attention_length_before_list = [args.attention_length_before]
    attention_length_after_list = [args.attention_length_after]
    nb_random_sample = int(args.nb_random_sample)
    try:
        dep_relations_dict = read_yaml('/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/oldstuff/dependency_parsing/dependency_relations.yml')
    except:
        dep_relations_dict = {}

    output_attentions = False
    output_hidden_states = True
    same_freq = args.same_freq
    same_syntax = args.same_syntax
    print(same_syntax)

    #### Computations ####

    for index, bert_model in enumerate(pretrained_bert_models):
        extractor = BertExtractor(bert_model, 
                                language, 
                                names[index], 
                                prediction_types[index], 
                                output_hidden_states=output_hidden_states, 
                                output_attentions=output_attentions, 
                                attention_length_before=attention_length_before_list[index],
                                attention_length_after=attention_length_after_list[index],
                                config_path=config_paths[index], 
                                max_length=512, 
                                number_of_sentence=number_of_sentence_list[index], 
                                number_of_sentence_before=number_of_sentence_before_list[index], 
                                number_of_sentence_after=number_of_sentence_after_list[index],
                                )
        print(extractor.name, ' - Extracting activations ...')
        for run_index, iterator in enumerate(iterator_list):
            iterators = []
            text = iterator_list[run_index]
            if nb_random_sample > 0:
                if same_syntax and same_freq:
                    samples = [generate_pos_freq_sample(nlp, sentence, dep_relations_dict, n_samples=nb_random_sample, limit_iterations=3000, information_type='tag', use_morph=True, same_freq=True, word_freq=word_freq, skip_punctuation=True) for sentence in text]
                elif same_freq:
                    samples = [generate_freq_sample(sentence, n_samples=nb_random_sample, word_list=word_list, word_freq=word_freq, skip_punctuation=True) for sentence in text]
                iterators = list(zip(*samples))
                iterators = [[sent.lower() for sent in text] for text in iterators]
            else:
                iterators = [[sent.lower() for sent in iterator_list[run_index]]]

            gc.collect()
            print("############# Run {} #############".format(run_index + 1))
            hidden_states_activations = []
            attention_heads_activations = []
            for iter_ in tqdm(iterators):
                activations  = extractor.extract_activations(iter_, language)
                hidden_states_activations.append(activations[0].values)
                #attention_heads_activations.append(activations[1].values)
            hidden_states_activations = pd.DataFrame(np.mean(np.stack(hidden_states_activations, axis=0), axis=0), columns=activations[0].columns)
            #attention_heads_activations = pd.DataFrame(np.mean(np.stack(attention_heads_activations, axis=0), axis=0), columns=activations[1].columns)
            #(cls_hidden_states_activations, cls_attention_activations) = activations[2]
            #(sep_hidden_states_activations, sep_attention_activations) = activations[3]
            #activations = pd.concat([hidden_states_activations, attention_heads_activations], axis=1)
            #cls_activations = pd.concat([cls_hidden_states_activations, cls_attention_activations], axis=1)
            #sep_activations = pd.concat([sep_hidden_states_activations, sep_attention_activations], axis=1)

            check_folder(saving_path_folders[index])
            hidden_states_activations.to_csv(os.path.join(saving_path_folders[index], 'activations_run{}.csv'.format(run_index + 1)), index=False)
            transform(
                hidden_states_activations, 
                saving_path_folders[index], 
                'activations', 
                run_index=run_index,
                n_layers_hidden=extractor.config['num_hidden_layers'] + 1 if output_hidden_states else 0,
                n_layers_attention=extractor.config['num_hidden_layers'] if output_attentions else 0, 
                hidden_size=extractor.config['hidden_size'])
            #transform(cls_activations, saving_path_folders[index], 'cls')
            #transform(sep_activations, saving_path_folders[index], 'sep')
            
            #activations.to_csv(os.path.join(saving_path_folders[index], 'activations_run{}.csv'.format(run_index + 1)), index=False)
            #cls_activations.to_csv(os.path.join(saving_path_folders[index], 'cls_run{}.csv'.format(run_index + 1)), index=False)
            #sep_activations.to_csv(os.path.join(saving_path_folders[index], 'sep_run{}.csv'.format(run_index + 1)), index=False)
            del activations
            #del cls_activations
            #del sep_activations
            del hidden_states_activations
            del attention_heads_activations
            #del cls_hidden_states_activations
            #del cls_attention_activations
            #del sep_hidden_states_activations
            #del sep_attention_activations
            
