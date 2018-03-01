from bow_model import BowModel

import util # need this to initialize global vars
from util import get_all_samples_from_adir, encode
import os

MEANS_FILE = ''  # A dictionary from author name to mean vectors
DEV_DIR = '../data/val'
TEST_DIR = '../data/test'

correct = 0
incorrect = 0


def evaluate(d=DEV_DIR):
    global correct
    global incorrect
    
    model = BowModel()
    
    for author in os.listdir(d):
        if author[0] != '.':
            samples = get_all_samples_from_adir(author, datadir=DEV_DIR)

            for s in samples:
                s_encoding = encode(s)
                a = model.predict(s_encoding)
                if a == author:
                    correct += 1
                else:
                    incorrect += 1
    print('Correct: %d' % correct)
    print('Incorrect: %d' % incorrect)

if __name__ == "__main__":
    evaluate()
