import numpy as np
import os
from os.path import join
import pickle

TRAIN_DIR = '../data/train'
VAL_DIR = '../data/val'
TEST_DIR = '../data/test'

PUNCS = '.,?!\'":;-()'


def get_puncs_map():
    puncs = {c: i+1 for i, c in enumerate(PUNCS)}
    puncs['%'] = 0
    with open('../puncs_map.pkl', 'wb') as f:
        pickle.dump(puncs, f)
    return puncs


def get_authors_map():
    def is_author(name):
        return ''.join(name.split('_')).isalpha()
    authors = [a for a in os.listdir(TRAIN_DIR) if is_author(a)]
    authors.sort()
    authors_map = {a: i for i, a in enumerate(authors)}
    with open('../authors_map.pkl', 'wb') as f:
        pickle.dump(authors_map, f)
    return authors, authors_map


def read_data(puncs_map, authors, authors_map, d=TRAIN_DIR):
    all_tokens = []
    all_authors = []
    for author in authors:
        print('Reading from author %s' % author)
        for f in os.listdir(join(d, author)):
            if f.endswith('.pkl'):
                tokens = pickle.load(open(join(d, author, f), 'rb'))
                tokens = map(puncs_map.get, tokens)
                all_tokens.append(tokens)
                author_vec = np.zeros(len(authors))
                author_vec[authors_map[author]] = 1
                all_authors.append(author_vec)
    return (np.asarray(all_tokens),
            np.asarray(all_authors))


def load_data(split='train'):
    if split == 'train':
        d = TRAIN_DIR
    if split == 'val':
        d = VAL_DIR
    if split == 'test':
        d = TEST_DIR
    return (pickle.load(open(join(d, 'tokens.pkl'), 'rb')),
            pickle.load(open(join(d, 'authors.pkl'), 'rb')))


if __name__ == '__main__':
    p_map = get_puncs_map()
    a, a_map = get_authors_map()
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        tokens, authors = read_data(p_map, a, a_map, d)
        pickle.dump(tokens, open(join(d, 'tokens.pkl'), 'wb'))
        pickle.dump(authors, open(join(d, 'authors.pkl'), 'wb'))
