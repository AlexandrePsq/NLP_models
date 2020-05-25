from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
import os
import re
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


def tokenize(path, language, train=False):
    """ Tokenize a text into sentences.
    Optionnaly preprocess it.
    Arguments:
        - path: (str) path or text
        - language: (str)  
    Returns:
        - iterator: sentence iterator
    """
    print('Tokenizing...')
    if os.path.exists(path):
        path = open(path, 'r', encoding='utf8').read()

    if not train:
        print('Preprocessing...')
        text = preprocess(path, special_words, language)
        print('Preprocessed.')
    else:
        text = path
    # iterator = [unk_transform(item, vocab).lower() for item in text.split()]
    iterator = [item.strip() for item in tqdm(text.split('\n')[:-1])] # vocab words not lowered
    print('Tokenized.')
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
    
    ### tokenize without punctuation ###
    # for item in punctuation:
    #     text = text.replace(item, ' ')
    ### tokenize with punctuation ###
    # ### tokenize thanks to usual tools for text without strange characters ###
    # tokenized = sent_tokenize(text, language=language)
    # tokenized = [word_tokenize(sentence, language=language) + ['<eos>'] for sentence in tokenized]
    # iterator = [unk_transform(item, vocab).lower() for sublist in tokenized for item in sublist]
    return text