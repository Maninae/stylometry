import numpy as np
import os
from os.path import join
import pickle

TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'
TEST_DIR = '../data/test'

PUNCS = '.,?!\'":;-()'

def read_data(d=TRAIN_DIR):
    def is_author(name):
        return ''.join(name.split('_')).isalpha()
    puncs = {c: i+1 for i, c in enumerate(PUNCS)}
    puncs['%'] = 0
    with open('../puncs_map.pkl', 'wb') as f:
        pickle.dump(puncs, f)
    all_tokens = []
    all_authors = []
    authors = [a for a in os.listdir(d) if is_author(a)]
    authors.sort()
    authors_map = {a: i for i, a in enumerate(authors)}
    with open('../authors_map.pkl', 'wb') as f:
        pickle.dump(authors_map, f)
    for author in authors:
        if is_author(author):
            print('Reading from author %s' % author)
            for f in os.listdir(join(d, author)):
                if f.endswith('.pkl'):
                    tokens = pickle.load(open(join(d, author, f)))
                    tokens = map(puncs.get, tokens)
                    all_tokens.append(tokens)
                    all_authors.append(authors_map[author])
    return (np.asarray(all_tokens),
            np.asarray(all_authors)[:, None],
            authors_map)
