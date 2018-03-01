import pickle
import numpy as np

from os.path import join
from os import listdir


__NUM__ = "__NUM__"
traindir = '../data/train'

__SYMBOLS__ = ['.', ',', '?', '!', '\'', '"', ':', ';', '-', '(', ')', __NUM__]

__symbol_to_idx = {}
for i, s in enumerate(__SYMBOLS__):
    symbol_map[s] = i

def encode(text):
    text = text.split()
    vec = np.zeros((len(__SYMBOLS__),)) # 1D zero vector
    for w in text:
        
        if w.isdigit():
            idx = __symbol_to_idx[__NUM__]
            cnt = int(w)
        else:
            idx = __symbol_to_idx[w]
            cnt = 1

        vec[idx] += cnt
    return vec


def open_pickle(filepath):
    content = pickle.load(filepath)
    return content


def get_all_samples_from_adir(authorname):
    adir = join(traindir, authorname)
    filenames = [fn for fn in listdir(adir) if fn[-4:] == '.pkl']
    samples = [open_pickle(join(adir, fn)) for fn in filenames]
    return samples

