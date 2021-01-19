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
        'redfaced': 'red faced',
    },
    'french': {

    }
}

words2replace = {'english': {
        'primeval': 'primitive',
        'boa': 'snake',
        'constrictor': 'constricted',
        'Boa': 'snake',
        'constrictors': 'constricted',
        'grownups': 'grown-up',
        'digesting': 'digest',
        'tiresome': 'tiring',
        've': 'have',
        'neckties': 'ties',
        'grownup': 'grown-up',
        'readied': 'ready',
        'hadn\'t': 'had not',
        'hadn‘t': 'had not',
        'stared': 'stare',
        'eyed': 'looked',
        'disobey': 'disobeyed',
        'Absurd': 'absurd',
        'crossly': 'across',
        'redrew': 'drew',
        'astounded': 'astounding',
        'indulgently': 'leniency',
        'redid': 'did', 
        'peal': 'peal',
        'daydream': 'dream',
        'll': 'will',
        'Tie': 'tie',
        'Anywhere': 'anywhere',
        'Asteroid': 'asteroid',
        'stylishly': 'style',
        'geraniums': 'plants',
        'shrug': 'bow',
        'alright': 'OK',
        'baobabs': 'trees',
        'baobab': 'tree',
        'elongates': 'elongate',
        'timidly': 'timid',
        'sprig': 'twigs',
        'radish': 'vegetable',
        'pierces': 'penetrates',
        'moralist': 'Puritain',
        'skirting': 'skirt',
        'sunsets': 'sunset',
        'unscrew': 'detach',
        'frightful': 'horrible',
        'budge': 'move',
        'dumbfounded': 'astonished',
        'smelled': 'smell',
        'sprouted': 'germinate',
        'sprout': 'germinate',
        'rumpled': 'crushed',
        'adornment': 'embellishments',
        'yawn': 'deep',
        'woken': 'wake',
        'presentable': 'tidy',
        'abashed': 'embarrassed',
        'sprinkling': 'scattering',
        'detest': 'hate',
        'Detests': 'hates',
        'Humiliated': 'humiliated',
        'coughed': 'cough',
        'perfumed': 'perfume',
        'confidences': 'confidence',
        'contrivances': 'scheme',
        'Volcanic': 'volcanic',
        'reproaches': 'lecture',
        'naively': 'naive',
        'linger': 'lingered',
        'yawned': 'deep',
        'yawning': 'deep',
        'Yawns': 'deep',
        'frightens': 'frighten',
        'blushing': 'colouring',
        'Hum': 'hum',
        'sputtered': 'faltered',
        'vexed': 'annoyed',
        'seabird': 'bird',
        'enquired': 'enquiry',
        'majestically': 'majestic',
        'Sire': 'sire',
        'forsaken': 'abandoned',
        'almanac': 'calendar',
        'regretting': 'regret',
        'leant': 'bend',
        'grieve': 'grief',
        'conceited': 'narcissistic',
        'Clap': 'clap',
        'clapped': 'applause',
        'monotony': 'monotone',
        'Admire': 'admire',
        'shrugging': 'bowing',
        'gloomily': 'gloom',
        'Ashamed': 'ashamed',
        'Phew': 'hum',
        'Huh': 'hum',
        'balderdash': 'nonsense',
        'grumpily': 'crank',
        'lamplighter': 'lighter',
        'saluted': 'salutes',
        'sponged': 'sponge',
        'sigh': 'sight',
        'panted': 'puff',
        'glanced': 'looked',
        'strolling': 'stroll',
        'prettiest': 'pretty',
        'lamplighters': 'lighters',
        'adore': 'love',
        'chore': 'chores',
        'courteously': 'courteous',
        'Onto': 'onto',
        'homesick': 'sick',
        'nondescript': 'ordinary',
        'footstool': 'stool',
        'tamed': 'tame',
        'bothersome': 'annoying',
        'sighed': 'sighted',
        'hurrying': 'hurry',
        'gazed': 'looked',
        'tames': 'tame',
        'grumble': 'complain',
        'switchman': 'man',
        'rumbling': 'echo',
        'thundered': 'thunder',
        'squashing': 'crushing',
        'quench': 'extinguish',
        'saver': 'investor',
        'Myself': 'myself',
        'weariness': 'weary',
        'immensity': 'immense',
        'feverish': 'burning',
        'enchantment': 'witchcraft',
        'trembled': 'shake',
        'moaned': 'complained',
        'weathervane': 'vane',
        'trembling': 'shaking',
        'shimmer': 'flash',
        'cabbages': 'vegetables',
        'ok': 'OK',
        'pang': 'pain',
        'blushed': 'coloured',
        'hesitantly': 'hesitant',
        'blush': 'colour',
        'muffler': 'silence',
        'moistened': 'dampen',
        'plummeting': 'plummeted',
        'Snakes': 'snakes',
        'determinedly': 'determined',
        'fasten': 'fastened',
        'Surely': 'surely',
        'sweetly': 'sweet',
        'Herein': 'here',
        'yourselves': 'yourself',
        "hasn ’ t": "has ’ n’t", 
        "hadn ’ t": "had ’ n’t", 
        "didn ’ t": "did ’ n’t", 
        "doesn ’ t": "does ’ n’t", 
        "isn ’ t": "is ’ n’t", 
        "Grownups": "grown-up", 
        "wouldn ’ t": "would ’ n’t", 
        "aren ’ t": "are ’ n’t", 
        "wasn ’ t": "was ’ n’t", 
        "tidy": "clean", 
        "shouldn ’ t": "should ’ n’t", 
        "Weren ’ t": "were ’ n’t", 
        "couldn ’ t": "could ’ n’t", 
        "mustn ’ t": "must ’ n’t", 
    
        "hasn ' t": "has ’ n’t", 
        "hadn ' t": "had ’ n’t", 
        "didn ' t": "did ’ n’t", 
        "doesn ' t": "does ’ n’t", 
        "isn ' t": "is ’ n’t", 
        "wouldn ' t": "would ’ n’t", 
        "aren ' t": "are ’ n’t", 
        "wasn ' t": "was ’ n’t", 
        "shouldn ' t": "should ’ n’t", 
        "Weren ' t": "were ’ n’t", 
        "couldn ' t": "could ’ n’t", 
        "mustn ' t": "must ’ n’t", 
        "Herein": "here", 
}
                }


def tokenize(path, language, train=False, vocab=None):
    """ Tokenize a text into sentences.
    Optionnaly preprocess it.
    Arguments:
        - path: (str) path or text
        - language: (str)  
    Returns:
        - iterator: word iterator
    """
    if os.path.exists(path):
        path = open(path, 'r', encoding='utf8').read()

    if not train:
        text = preprocess(path, special_words, language)
    else:
        text = path
    # iterator = [unk_transform(item, vocab).lower() for item in text.split()]
    iterator_ = [item for item in tqdm(text.split('\n')[:-1])] # vocab words not lowered
    iterator = [unk_transform(word, vocab) for item in tqdm(iterator_) for word in item.strip().split(' ')]
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
        text = text.replace(' ' + word + ' ', ' ' + special_words[language][word] + ' ')
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
    #for word in words2replace[language].keys():
    #    text = text.replace(' ' + word + ' ', ' ' + words2replace[language][word] + ' ')
    
    ### tokenize without punctuation ###
    # for item in punctuation:
    #     text = text.replace(item, ' ')
    ### tokenize with punctuation ###
    # ### tokenize thanks to usual tools for text without strange characters ###
    # tokenized = sent_tokenize(text, language=language)
    # tokenized = [word_tokenize(sentence, language=language) + ['<eos>'] for sentence in tokenized]
    # iterator = [unk_transform(item, vocab).lower() for sublist in tokenized for item in sublist]
    return text


def unk_transform(word, vocab=None):
    if word == 'unk':
        return '<unk>'
    elif not vocab:
        return word
    elif word in vocab.idx2word:
        return word
    else:
        print(word)
        return '<unk>'

