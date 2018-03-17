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

def preprocess(input_tensor):
    """
    args:
      input_tensor: (None, input_length) tensor of ints
    return:
      (None, input_length, num_classes=_NUM_TOKENS=12) tensor of one-hots
    """
    #embeddings = 
    return K.one_hot(input_tensor, _NUM_TOKENS)
    #return embeddings