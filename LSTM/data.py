import os
import numpy as np
import torch
#from .tokenizer import tokenize
from collections import defaultdict
import logging
from tqdm import tqdm

#------------------------------------------------------------------------
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
        'hasn‘t': 'hasnt'
    },
    'french': {

    }
}


def tokenize(path, language, vocab=None, path_like=True, train=False):
    print('Tokenizing...')
    if path_like:
        assert os.path.exists(path)
        path = open(path, 'r', encoding='utf8').read()

    if not train:
        print('Preprocessing...')
        text = preprocess(path, special_words, language)
        print('Preprocessed.')
    else:
        text = path
    # iterator = [unk_transform(item, vocab).lower() for item in text.split()]
    iterator = [unk_transform(item, vocab) for item in tqdm(text.split())] # vocab words not lowered
    print('Tokenized.')
    return iterator


def unk_transform(word, vocab=None):
    if word == 'unk':
        return '<unk>'
    elif not vocab:
        return word
    elif word in vocab.idx2word:
        return word
    else:
        return '<unk>'


def preprocess(text, special_words, language):
    text = text.replace('\n', '')
    text = text.replace('<unk>', 'unk')
    for word in special_words[language].keys():
        text = text.replace(word, special_words[language][word])
    transf = inflect.engine()
    numbers = re.findall('\d+', text)
    for number in numbers:
        text = text.replace(number, transf.number_to_words(number))
    punctuation = ['.', '\'', ',', ';', ':', '!', '?', '/', '-', '"', '‘', '’', '(', ')', '{', '}', '[', ']', '`', '“', '”', '—']
    for item in punctuation:
        text = text.replace(item, ' '+ item + ' ')
    text = text.replace('.  .  .', '...')
    ### tokenize without punctuation ###
    # for item in punctuation:
    #     text = text.replace(item, ' ')
    ### tokenize with punctuation ###
    # ### tokenize thanks to usual tools for text without strange characters ###
    # tokenized = sent_tokenize(text, language=language)
    # tokenized = [word_tokenize(sentence, language=language) + ['<eos>'] for sentence in tokenized]
    # iterator = [unk_transform(item, vocab).lower() for sublist in tokenized for item in sublist]
    return text

#------------------------------------------------------------------------

class Dictionary(object):
    def __init__(self, path, language, max_size=50000):
        self.word2idx = {}
        self.idx2word = []
        self.max_size = max_size
        self.language = language
        self.word2freq = defaultdict(int)

        vocab_path = os.path.join(path, 'vocab.txt')
        try:
            vocab = open(vocab_path, encoding="utf8").read()
            self.word2idx = {w: i for i, w in enumerate(vocab.split('\n'))}
            self.idx2word = [w for w in vocab.split('\n')]
            self.vocab_file_exists = True
        except FileNotFoundError:
            logging.info("Vocab file not found, creating new vocab file.")
            self.create_vocab(os.path.join(path, 'train.txt'))
            logging.info("Saving...")
            assert len(self.idx2word)==len(self.word2idx.keys())
            open(vocab_path,"w").write("\n".join([w for w in self.idx2word]))


    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2freq[word] = 1
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        else:
            self.word2freq[word] += 1

    def filter_by_freq(self, threshold):
        pass
    
    def stop_at_max_size(self):
        if '<unk>' in self.word2idx.keys():
            self.max_size += 1
        freqs = np.array([self.word2freq[w] for w in self.idx2word])
        order = np.argsort(freqs)
        unkwown = list(np.array(self.idx2word)[order][:max(-self.max_size, -len(self.idx2word))])
        self.idx2word = list(np.array(self.idx2word)[order][max(-self.max_size, -len(self.idx2word)):])
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        
    def __len__(self):
        return len(self.idx2word)


    def create_vocab(self, path):
        iterator = open(path, 'r').read().replace('\n', ' ')
        iterator = re.sub(' +', ' ', iterator).split(' ')
        for item in tqdm(iterator):
            self.add_word(item)
        logging.info(f"Filtering rare words to get {self.max_size}...")
        self.stop_at_max_size()
        self.add_word('<unk>')



class Corpus(object):
    def __init__(self, path, language):
        print('Building dictionary...')
        self.dictionary = Dictionary(path, language)
        print('Dictionary built.')
        train_path = os.path.join(path, 'train.txt')
        valid_path = os.path.join(path, 'valid.txt')
        test_path = os.path.join(path, 'test.txt')
        train_tensor = os.path.join(path, 'train.pkl')
        valid_tensor = os.path.join(path, 'valid.pkl')
        test_tensor = os.path.join(path, 'test.pkl')
        try:
            with open(train_tensor, 'rb') as f:
                self.train = torch.load(f)
        except FileNotFoundError:
            logging.info("Tensor files not found, creating new tensor files.")
            print('Computing train tensor...')
            train_data = open(train_path, 'r').read().replace('\n', ' ')
            train_data = re.sub(' +', ' ', train_data).split(' ')
            self.train = create_tokenized_tensor(train_data, self.dictionary)
            print('Train tensor computed.')
            with open(train_tensor, 'wb') as f:
                torch.save(self.train, f)
        try:
            with open(valid_tensor, 'rb') as f:
                self.valid = torch.load(f)
        except FileNotFoundError:
            logging.info("Tensor files not found, creating new tensor files.")
            print('Computing valid tensor...')
            valid_data = open(valid_path, 'r').read().replace('\n', ' ')
            valid_data = re.sub(' +', ' ', valid_data).split(' ')
            self.valid = create_tokenized_tensor(valid_data, self.dictionary)
            print('Valid tensor computed.')
            with open(valid_tensor, 'wb') as f:
                torch.save(self.valid, f)
        try:
            with open(test_tensor, 'rb') as f:
                self.test = torch.load(f)
        except FileNotFoundError:
            logging.info("Tensor files not found, creating new tensor files.")
            print('Computing test tensor...')
            test_data = open(test_path, 'r').read().replace('\n', ' ')
            test_data = re.sub(' +', ' ', test_data).split(' ')
            self.test = create_tokenized_tensor(test_data, self.dictionary)
            print('Test tensor computed.')
            with open(test_tensor, 'wb') as f:
                torch.save(self.test, f)
        



def create_tokenized_tensor(iterator, dictionary):
    """Create tensor of embeddings from word iterator."""
    tensor = torch.LongTensor(len(iterator))
    token = 0
    for item in tqdm(iterator):
        tensor[token] = dictionary.word2idx[item] if item in dictionary.word2idx else dictionary.word2idx['<unk>']
        token += 1
    return tensor
