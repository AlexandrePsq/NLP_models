import os
import wget
import time
import yaml
import glob
import torch
import random
import inspect
import datetime
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, GPT2Config, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, WEIGHTS_NAME, CONFIG_NAME

#########################################
############ Basic functions ############
#########################################

def check_folder(path):
    """Create adequate folders if necessary."""
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass
    
def read_yaml(yaml_path):
    """Open and read safely a yaml file."""
    with open(yaml_path, 'r') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except :
            print("Couldn't load yaml file: {}.".format(yaml_path))
    return parameters

def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Arguments:
        - data: list/dict/str/int/float
        -yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def filter_args(func, d):
    """ Filter dictionary keys to match the function arguments.
    Arguments:
        - func: function
        - d: dict
    Returns:
        - args: dict
    """
    keys = inspect.getfullargspec(func).args
    args = {key: d[key] for key in keys if ((key!='self') and (key in d.keys()))}
    return args

def get_device(device_number=0, local_rank=-1):
    """ Get the device to use for computations.
    """
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        if torch.cuda.is_available():
            print('We will use the GPU:', torch.cuda.get_device_name(device_number))
        else:
            print('No GPU available, using the CPU instead.')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def set_seed(value=1111):
    """ Set all seeds to a given value for reproductibility."""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


#########################################
########## Specific functions ###########
#########################################

def save(model, tokenizer, output_dir, index):
    """ Saving best-practices: if you use defaults names for the model, 
    you can reload it using from_pretrained().
    """
    output_dir = os.path.join(output_dir, index)
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)

def pick_random_word(words, vocabulary):
    """ Replace a word with another.
    Can be applied to list of words.
    Args:
        - words: str (or list of str)
        - vocabulary: list (of strings)
    Returns:
        - new_words: str (or list of str)
    """
    new_words = []
    if type(words)==list:
        new_words = [random.sample(vocabulary, 1) for word in words]
    elif type(words)==str:
        new_words = random.sample(vocabulary, 1)
    return new_words


def pick_pos_word(words, dictionary):
    """ Replace a word with another with same POS tagging.
    Can be applied to list of words.
    Args:
        - words: str (or list of str)
        - dictionary: dict {word: POS ; POS: [word1, ..., wordN]}
    Returns:
        - new_words: str (or list of str)
    """
    new_words = []
    if type(words)==list:
        for word in words:
            pos = dictionary[word]
            replacement = random.sample(dictionary[pos], 1)
            new_words.append(replacement)
    elif type(words)==str:
        new_words = random.sample(dictionary[dictionary[words]], 1)
    return new_words


def transform_context(sequence, transformation='shuffle', dictionary=None, vocabulary=None, start_at=None, stop_at=None):
    """ Shuffle all words in a sequence of tokens (words/punctuation signs), or replace them with 
    words with identical POS or random words.
    Args:
        - sequence: list of words/punctuation signs
        - start_at: int (start shuffling at this index)
    Returns:
        - new_sequence: list of words/punctuation signs
    """ 
    words = sequence.split()
    punctuation = ['.', '!', '?', '...', '\'', ',', ';', ':', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—']
    if stop_at is not None:
        supp = len([word for word in words[stop_at:] if word in punctuation]) # we do not count punctuation in the number of words to shuffle
        stop_at -= supp
    else:
        stop_at = np.inf
    if start_at is not None:
        supp = len([word for word in words[:start_at] if word in punctuation]) # we do not count punctuation in the number of words to shuffle
        start_at += supp
    else:
        start_at = np.inf
    
    # For each word, we compute the index of the other words to shuffle
    index_words = [i for i, item in enumerate(words) if item not in punctuation if ((i > start_at) or (i <= stop_at))]
    new_sequence = np.array(words.copy())
    if transformation=='shuffle':
        new_words_index = random.sample(index_words, len(index_words))
        new_sequence[index_words] = new_sequence[new_words_index]
    elif transformation=='pos_replacement':
        new_sequence[index_words] = pick_pos_word(new_sequence[index_words], dictionary)
    elif transformation=='random_replacement':
        new_sequence[index_words] = pick_random_word(new_sequence[index_words], vocabulary)
    return new_sequence


def transform_sentence_and_context(
    iterator, 
    past_context_size, 
    pretrained_model,
    transformation='shuffle',
    vocabulary=None,
    dictionary=None,
    select=None,
    seed=1111,
    add_prefix_space=True):
    """ Given a list of sentences, for each word, we transform its context outside a certain context window.
    Args:
        - iterator: list (of str)
        - context_size: int
        - pretrained_model: str
        - vocabulary: list
        - dictionary: dict
        - seed: int
        - add_prefix_space: bool
    Returns:
        - batch_tmp: list (of str)
        - index_tmp: list (of tuple of int)
    """
    random.seed(seed)
    punctuation = ['.', '!', '?', '...', '\'', ',', ';', ':', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—']
    if select is None:
        words = ' '.join(iterator).split()
    else:
        words = iterator[select].split()
        
    all_words = ' '.join(iterator).split()
    words_before = [] if select is None else ' '.join(iterator[:select]).split()
    supp_before = [len([word for word in all_words[max(j+len(words_before)+1-past_context_size, 0):j+len(words_before)+1] if word in punctuation]) for j in range(len(words))] # we do not count punctuation in the number of words to shuffle

    # For each word, we compute the index of the other words to transform
    # We transform past context. Change conditions "i<j" and ... to something else if needed
    index_words_list_before = [[i for i, item in enumerate(all_words) if item not in punctuation if ((i!=(j+len(words_before))) and  (i <= j+len(words_before)-past_context_size-supp_before[j]))] for j in range(len(words))] # '<=' because context_size of 1 is the current word

    # Create the new array of sentences with original words 
    new_words = np.tile(np.array(all_words.copy()), (len(words), 1))

    for i in range(len(new_words)):
        if len(index_words_list_before[i])>0: # if there are words to change...
            if transformation=='shuffle':
                # Replace words that need to be shuffled by the random sampling (except fix point and punctuation)
                new_order = random.sample(index_words_list_before[i], len(index_words_list_before[i]))
                if len(index_words_list_before[i])>1:
                    while new_order==index_words_list_before[i]:
                        new_order = random.sample(index_words_list_before[i], len(index_words_list_before[i]))
                new_words[i, index_words_list_before[i]] = new_words[i, new_order]
            elif transformation=='pos_replacement':
                # Replace words that need to be replaced by words with same POS (except fix point and punctuation)
                new_words[i, index_words_list_before[i]] = pick_pos_word(new_words[i, index_words_list_before[i]], dictionary)
            elif transformation=='random_replacement':
                # Replace words that need to be replaced by random words (except fix point and punctuation)
                new_words[i, index_words_list_before[i]] = pick_random_word(new_words[i, index_words_list_before[i]], vocabulary)

    # Convert array to list
    new_words = list(new_words)
    new_words = [list(item) for item in new_words]
    batch_tmp = []
    index_tmp = []
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model) # to replace with tokenizer of interest
    # adding transformed context to each sentence
    for i, sentence in enumerate(new_words):
        batch_tmp.append(' '.join(sentence).strip())
        # Determining associated indexes
        tmp1 = ' '.join(sentence[:i+len(words_before)])
        tmp2 = ' '.join(sentence[:i+len(words_before)+1])
        index_tmp.append((len(tokenizer.tokenize(tmp1.strip(), add_prefix_space=add_prefix_space)), 
                     len(tokenizer.tokenize(tmp2.strip(), add_prefix_space=add_prefix_space))
                    )) # to replace with tokenizer of interest and arguments
    return batch_tmp, index_tmp

def batchify_with_detailed_indexes(iterator, number_of_sentence, number_sentence_before, pretrained_gpt2, max_length=512, stop_attention_at_sent=None, stop_attention_before_sent=0, add_prefix_space=True):
    """Batchify iterator sentence, to get batches of specified number of sentences.
    Arguments:
        - iterator: sentence iterator
        - number_of_sentence: int
        - number_sentence_before: int
    Returns:
        - batch: sequence iterator
        - indexes: tuple of int
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 3 # for special tokens, because there is a G (dot) before last special token
    assert number_of_sentence > 0
    if stop_attention_before_sent > 0:
        stop_attention_at_sent += 1
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_gpt2)
    
    batch = []
    indexes = []
    sentence_count = 0
    n = len(iterator)
    stop = 0

    if number_sentence_before > 0:
        start = 0
        if stop_attention_at_sent is not None:
            while stop < number_sentence_before:
                stop = min(start + stop_attention_at_sent + number_of_sentence, n)
                token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=add_prefix_space))
                if token_count > max_length:
                    raise ValueError('Cannot fit context with additional sentence. You should reduce context length.')
                batch.append(' '.join(iterator[start:stop]))
                beg = 0
                res = []
                for item in iterator[start:stop]:
                    end = len(tokenizer.tokenize(item, add_prefix_space=add_prefix_space)) + beg
                    res.append((beg, end))
                    beg = end
                indexes.append(res)
                start += 1
            sentence_count = stop
            
        else:
            stop = min(number_sentence_before, n)
            token_count = len(tokenizer.tokenize(' '.join(iterator[:stop]), add_prefix_space=add_prefix_space))
            if token_count > max_length:
                raise ValueError('Cannot fit context with additional sentence. You should reduce context length.')
            batch.append(' '.join(iterator[:stop]))
            beg = 0
            res = []
            for item in iterator[:stop]:
                end = len(tokenizer.tokenize(item, add_prefix_space=add_prefix_space)) + beg
                res.append((beg, end))
                beg = end
            indexes.append(res)
            sentence_count = stop

    while sentence_count < n:
        start = sentence_count - number_sentence_before
        stop = min(sentence_count + number_of_sentence, n)
        token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=add_prefix_space))
        if token_count > max_length:
            raise ValueError('Too many context sentence. You reach {} tokens only with context.'.format(token_count))
        batch.append(' '.join(iterator[start:stop]))
        beg = 0
        res = []
        for item in iterator[start:stop]:
            end = len(tokenizer.tokenize(item, add_prefix_space=add_prefix_space)) + beg
            res.append((beg, end))
            beg = end
        indexes.append(res)
        sentence_count = stop
    return batch, indexes

def batchify_text_with_memory_size(iterator, tokenizer, memory_size, bos='<|endoftext|>', eos='<|endoftext|>'):
    """Create batch of identical size of truncated input with given size in number of words.
    Arguments:
        - iterator: sentence iterator
        - tokenizer: tokenizer instance
        - memory_size: int
    Returns:
        - final_iterator: list of arrays of int
        - final_mappings: array of dict
    """
    iterator = (' '.join(iterator)).split()
    iterator_tmp = []
    iterator_tmp.append([bos] + iterator[:memory_size] + [eos])
    for index, _ in enumerate(iterator[memory_size:]):
        iterator_tmp.append([bos] + iterator[1 + index:1 + index+memory_size] + [eos])
    mapping_tmp = [match_tokenized_to_untokenized(tokenizer.tokenize(' '.join(sent), add_prefix_space=False), ' '.join(sent)) for sent in iterator_tmp]
    tokenized_inputs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(sent), add_prefix_space=False)) for sent in iterator_tmp]
    final_iterator = [np.array(tokenized_inputs[0:1])]
    final_mappings = [np.array(mapping_tmp[0:1])]
    size = len(tokenized_inputs[1])
    start = 1
    stop = 2
    while stop < len(tokenized_inputs):
        if len(tokenized_inputs[stop])!=size:
            final_iterator.append(np.array(tokenized_inputs[start:stop]))
            final_mappings.append(np.array(mapping_tmp[start:stop]))
            start = stop
            size = len(tokenized_inputs[stop])
        stop += 1
    if (stop-start) > 1:
        final_iterator.append(np.array(tokenized_inputs[start:stop]))
        final_mappings.append(np.array(mapping_tmp[start:stop]))
    
    return final_iterator, final_mappings

def batchify_sentences(
    iterator,
    number_of_sentence, 
    number_sentence_before, 
    pretrained_model,
    past_context_size,
    transformation,
    vocabulary=None,
    dictionary=None,
    seed=1111,
    max_length=512,
    add_prefix_space=True,
    **kwargs
):
    """Prepare text to be processed by the model, applying either shuffling, randomization or replacement outside the context window..
    Arguments:
        - iterator: sentence iterator
        - number_of_sentence: int
        - number_sentence_before: int
        - pretrained_model: str
        - past_context_size: int
        - transformation: int 
        - vocabulary: list (or something else ?)
        - dictionary: dict
        - seed: int
        - max_length: int
        - add_prefix_space: bool
        
    Returns:
        - batches: list of str
        - indexes: list of tuples
    """
    iterator = [item.strip() for item in iterator]
    max_length -= 3 # for special tokens, because there is a G (dot) before last special token
    assert number_of_sentence > 0
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    batch = []
    indexes = []
    sentence_count = 0
    n = len(iterator)
    
    # rest of the iterator + context 
    while sentence_count < n:
        start = max(sentence_count - number_sentence_before, 0)
        stop = min(sentence_count + number_of_sentence, n)

        token_count = len(tokenizer.tokenize(' '.join(iterator[start:stop]), add_prefix_space=add_prefix_space)) # to replace with tokenizer of interest and arguments
        if token_count > max_length:
            raise ValueError('Cannot fit context with additional sentence. You should reduce context length.')
        # computing batch and indexes
        batch_tmp, index_tmp = transform_sentence_and_context(
            iterator[start:stop], 
            past_context_size=past_context_size,
            pretrained_model=pretrained_model,
            transformation=transformation,
            vocabulary=vocabulary,
            dictionary=dictionary,
            select=stop-start-1,
            seed=seed,
            add_prefix_space=add_prefix_space,
            **kwargs
        )        
        batch += batch_tmp
        indexes += index_tmp
        
        sentence_count = stop
        
    return batch, indexes

#########################################
###### Activations related functions ####
#########################################


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

def extract_activations_from_token_activations(activation, mapping, indexes):
    """Take the average activations of the tokens related to a given word."""
    new_activations = []
    key = None
    for key_, value in mapping.items(): 
        if indexes[0] in value:
            key = key_ 
    for word_index in range(key, len(mapping.keys()) - 1):
        word_activation = []
        word_activation.append([activation[:,index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    #print(' '.join([tokenizer.decode(tokenizer.convert_tokens_to_ids([tokenized_text[word] for word in mapping[index]])) for index in range(key, len(mapping.keys()) - 1)]))

    return new_activations

def extract_heads_activations_from_token_activations(activation, mapping, indexes):
    """Extract heads activations of each layer for each token.
    Take the average activations of the tokens related to a given word.
    activation.shape: [nb_layers, nb_heads, sequence_length, hidden_size/nb_heads]"""
    new_activations = []
    key = None
    for key_, value in mapping.items(): 
        if indexes[0] in value:
            key = key_
    for word_index in range(key, len(mapping.keys()) - 2):
        word_activation = []
        word_activation.append([activation[:, :, index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        new_activations.append(np.mean(word_activation, axis=0).reshape(1,-1))
    return new_activations