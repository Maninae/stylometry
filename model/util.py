import keras.backend as K
from keras.utils import np_utils

# CONSTANTS
_NUM_AUTHORS = 11
_NUM_TOKENS = 12
_INPUT_LENGTH = 250

# EDIT: Unlikely to swap out any LSTM activations for this.
#   Can swish activate dense layers, but not deep enough.
def swish(x):
    return x * K.sigmoid(x)

custom_objects_dict = {
    'swish' : swish
}

def preprocess(x):
    """
    args:
      x: (None, input_length) ndarray of ints
    return:
      (None, input_length, num_classes=_NUM_TOKENS=12) ndarray, one hot
    """
    return np_utils.to_categorical(x, _NUM_TOKENS)

