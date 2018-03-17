from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Activation
# I think we need these, but not sure yet
from keras.layers import LSTMCell, RNN
from keras.models import Model
from keras.regularizers import l2

from keras import backend as K

from util import swish, custom_objects_dict
from util import _NUM_AUTHORS, _NUM_TOKENS, _INPUT_LENGTH # (11, 12, 250)


__testable_models = []

# Debug printing
def printl(obj, name):
    print("%s: %s" % (name, str(obj)))


# Decorator for keeping track of models to test
def testable(func):
    __testable_models.append(func)
    return func


def conv_model(embedding_dim=5):
    """ 12 tokens are transformed into a learned embedding.
        Conv2D

    """
    pass
    


@testable
def vanilla_LSTM_model():
    """ Standard LSTM model.
        Dense layers after output for more power, then classification.
    """
    return __LSTM_model(nb_lstm_units=256,
                        nb_dense_units=[128, 128],
                        lstm_dropout=0.25,
                        dense_dropout=0.25)

@testable
def stacked_LSTM_model():
    """ Two LSTMs stacked on top of one another, but learning less params each.
    """
    return __LSTM_model(nb_lstm_units=[128, 128],
                        nb_dense_units=[64, 64],
                        lstm_dropout=0.3,
                        dense_dropout=0.2,
                        return_sequences=True)


def __LSTM_model(nb_lstm_units,
                 nb_dense_units,
                 lstm_dropout,
                 dense_dropout,
                 dense_activation='relu',
                 output_average_state=False,
                 l2_reg=1e-3,
                 # layers.LSTM() parameters to pass on
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False):
    
    # input: (batch, input_length) tensor of ints (representing one-hot)
    main_input = Input(shape=(_INPUT_LENGTH,), dtype='int32', name='main_input')
    embeddings = K.one_hot(main_input, _NUM_TOKENS)
    x = embeddings

    # To use embeddings in Keras:
    # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # NOTE: we might not need to, with just 11 tokens.
    """
    x = Embedding(output_dim=10,            # dim of dense embedding
                  input_dim=_NUM_TOKENS,    # dim of one-hots
                  input_length=200          # length of sequence
                )(x)
    # output: (batch, input_length, output_dim)
    """
    # If we are doing stacked lstm, need return_sequences=True
    assert (type(nb_lstm_units) is list) == return_sequences
    
    if type(nb_lstm_units) is not list: # List of 1, if not a list
        nb_lstm_units = [nb_lstm_units]

    for nb_units in nb_lstm_units:
        print("Adding a LSTM layer with %d units, dropout %.03f."
              % (nb_units, lstm_dropout))

        x = LSTM(nb_units,
                 dropout=lstm_dropout,
                 kernel_regularizer=l2(l2_reg),
                 recurrent_regularizer=l2(l2_reg),
                 return_sequences=return_sequences,
                 return_state=return_state,
                 go_backwards=go_backwards)(x)
    # Remember an LSTM has multiple hidden states at each timestep.
    # Returns:
    #   if return_state: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each with shape (batch_size, nb_units).
    #   if return_sequences: 3D tensor with shape (batch_size, timesteps, nb_units).
    #   else, 2D tensor with shape (batch_size, nb_units).
    
    if output_average_state: # Intuition says this is prob. not necessary
        print("output_average_state=True. "
              "Output of LSTM is average of all timestep outputs.")
        if not return_sequences:
            raise ValueError("Must have output_average_state == return_sequences. "
                             "Nothing to aggregate if only last state returned.")
        else:
            # Reduce mean over dim [1], the timesteps.
            # NOTE: can work on attention, etc later to learn this aggregation. Not sure if necessary.
            x = K.mean(x, axis=1)

    # At this point, x.shape = (batch, nb_units)
    # Apply our list of dense layers
    if type(nb_dense_units) is not list: # List of 1, if not a list
        nb_dense_units = [nb_dense_units]

    for nb_units in nb_dense_units:
        print("Adding a dense layer with %d units, dropout %.03f, activation %s."
              % (nb_units, dense_dropout, dense_activation))

        x = Dense(nb_units, kernel_regularizer=l2(l2_reg))(x)
        x = Activation(dense_activation)(x)
        if dense_dropout > 0.0:
            x = Dropout(dense_dropout)(x)

    # Apply final dense layer to classify into 11 authors
    # NO SOFTMAX activation! The keras/tf softmax_crossentropy loss fn takes care of it.
    x = Dense(_NUM_AUTHORS)(x)

    printl(main_input, 'main_input')
    printl(x, 'x')
    return Model(inputs=main_input, outputs=x)


if __name__ == "__main__":
    print()
    print()
    
    for model_fn in __testable_models:
        print("Found model: %s." % model_fn.__name__)
        model = model_fn()
        model.summary()
