from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
import os
import re
import scipy.io.wavfile as wave
import numpy as np
import inflect
from tqdm import tqdm


special_words = {
    'english': {
        'grown-ups': 'grownups',
        'grown-up': 'grownup',
        'hasn\'t': 'hasnt',
        'hasn‘t': 'hasnt',
        'redfaced': 'red faced'
    },
    'french': {

    }
}

def sentence_to_words(iterator):
    iterator = [word for item in iterator for word in item.strip().split(' ')]
    return iterator


def tokenize(path, language, train=False, with_punctuation=False):
    """ Tokenize a text into sentences.
    Optionnaly preprocess it.
    Arguments:
        - path: (str) path or text
        - language: (str)  
    Returns:
        - iterator: sentence iterator (without punctuation)
    """
    punctuation = ['\'', ',', ';', ':', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—', '.', '!', '?']
    if os.path.exists(path):
        path = open(path, 'r', encoding='utf8').read()

    if not train:
        text = preprocess(path, special_words, language)
    else:
        text = path
    iterator = [item.strip() for item in tqdm(text.split('\n')[:-1])] # vocab words not lowered
    if not with_punctuation:
        for item in punctuation:
            iterator = [sentence.replace(item, '') for sentence in iterator]
    iterator = [re.sub(' +', ' ', sentence).strip() for sentence in iterator]
    iterator = [item for item in iterator if item!='']
    return iterator


def preprocess(text, special_words, language):
    """ Prepare text for tokenization into sentences.
    Replace words in text by the ones by which they have been replaced in the 
    textgrid files. Then replace all numbers by their written english version.
    We then add a space on each side of every punctuation symbol, paying attention 
    to '...'.
    Arguments:
        - text: (str) text to preprocess
        - special_words: (dict) special words and words by which to replace them
        - language: (str)
    """
    text = text.replace('\n', '')
    text = text.replace('<unk>', 'unk')
    for word in special_words[language].keys():
        text = text.replace(word, special_words[language][word])
    transf = inflect.engine()
    numbers = re.findall('\d+', text)
    for number in numbers:
        text = text.replace(number, transf.number_to_words(number))
    punctuation = ['\'', ',', ';', ':', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—']
    eos_punctuation =  ['.', '!', '?']
    for item in punctuation:
        text = text.replace(item, ' '+ item + ' ')
    text = text.replace('...', '<3 points>')
    for item in eos_punctuation:
        text = text.replace(item, ' '+ item + '\n')
    text = text.replace('<3 points>', ' ...\n')
    for item in eos_punctuation + ['...']:
        text = text.replace(item + '\n' + ' ' + '”', item + ' ' + '”' + '\n')
        text = text.replace(item + '\n' + ' ' + '’', item + ' ' + '’' + '\n')
    text = re.sub(' +', ' ', text)
    return text


def rms_tokenizer(path_to_audio, slice_period):
    # slice_period is in s
    # wave_file = wave.open(path, mode='r')
    # rate = wave_file.getframerate()
    # n_frames = wave_file.getnframes()   # Number of frames.
    # slice_length = int(slice_period * rate)
    [rate, data] = wave.read(path_to_audio)
    slice_length = int(slice_period * rate)
    data_list = [np.array(data[index * slice_length: index * slice_length + slice_length], dtype=np.float64) for index in range(len(data)//slice_length)]
    # Read audio data.
    # data = np.frombuffer(wave_file.readframes(n_frames), dtype=np.int16)
    # data_list = [data[index*slice_length: index*slice_length + slice_length] for index in range(n_frames//slice_length)]
    return data_list, rate, len(data), slice_length

