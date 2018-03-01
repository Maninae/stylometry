import numpy as np

__NUM__ = "__NUM__"

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


def text_of_file(filepath):
    content = ""
    with open(filepath, 'r') as f:
        content = f.readlines()


