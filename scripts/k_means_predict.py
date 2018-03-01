from baseline.bow_model import BowModel
from baseline.util import get_all_samples_from_adir
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
            samples = get_all_samples_from_adir(author, DEV_DIR)
            print samples
            for s in samples:
                a = model.predict(s)
                if a == author:
                    correct += 1
                else:
                    incorrect += 1
    print('Correct: %d' % correct)
    print('Incorrect: %d' % incorrect)


evaluate()
