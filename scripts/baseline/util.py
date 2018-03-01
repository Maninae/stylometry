import pickle
import numpy as np

from os.path import join
from os import listdir


NUM = "__NUM__"
traindir = '../data/train'

SYMBOLS = ['.', ',', '?', '!', '\'', '"', ':', ';', '-', '(', ')', NUM]

symbol_to_idx = {}
for i, s in enumerate(SYMBOLS):
    symbol_to_idx[s] = i
print("We have the symbol to idx mapping: %s" % str(symbol_to_idx))

def encode(text):
    text = text.split()
    vec = np.zeros((len(SYMBOLS),)) # 1D zero vector
    for w in text:

        if w.isdigit():
            idx = symbol_to_idx[NUM]
            cnt = int(w)
        else:
            idx = symbol_to_idx[w]
            cnt = 1

        vec[idx] += cnt
    return vec


def open_pickle(filepath):
    with open(filepath, 'rb') as f:
        content = pickle.load(f)
    return content


def get_all_samples_from_adir(authorname, datadir=traindir):
    adir = join(datadir, authorname)
    filenames = [fn for fn in listdir(adir) if fn[-4:] == '.pkl']
    samples = [open_pickle(join(adir, fn)) for fn in filenames]
    return samples

