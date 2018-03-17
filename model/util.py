import keras.backend as K

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

# Preprocessing functions, to change our puncs into one-hot ints
# https://keras.io/preprocessing/text/
