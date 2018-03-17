import pickle
from os.path import join
from keras.utils import np_utils

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'


def load_data(split='train'):
    if split == 'train':
        d = TRAIN_DIR
    if split == 'val':
        d = VAL_DIR
    if split == 'test':
        d = TEST_DIR
    return (pickle.load(open(join(d, 'tokens.pkl'), 'rb')),
            pickle.load(open(join(d, 'authors.pkl'), 'rb')))


def preprocess(x): # Expands to one hot
    """
    args:
      x: (None, input_length) ndarray of ints
    return:
      (None, input_length, num_classes=_NUM_TOKENS=12) ndarray, one hot
    """
    return np_utils.to_categorical(x, _NUM_TOKENS)


