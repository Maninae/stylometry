from keras.layers import Input, Embedding, LSTM, Conv1D, Dense, Dropout, Activation, Concatenate, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from keras import backend as K

from model.util import swish, custom_objects_dict
from model.util import _NUM_AUTHORS, _NUM_TOKENS, _INPUT_LENGTH # (11, 12, 250)


testable_models = []

# Debug printing
def printl(obj, name):
    print("%s: %s" % (name, str(obj)))


# Decorator for keeping track of models that are completed, ready to test
def testable(func):
    testable_models.append(func)
    return func
# Upon import of |stylometry_models|, `testable_models` will be populated.

######################################


@testable
def conv_model(embedding_dim=8,
               nb_filters=10,
               window_lengths=[7,11,15,19,23,27,31,35],
               nb_dense_units=64,
               conv_dropout=0.1,
               dense_dropout=0.2,
               l2_reg=1e-3):
    """ DO NOT PREPROCESS FOR THIS MODEL, we want the one-hot vecs as input.

        12 tokens are transformed into a learned embedding.
        Conv2D striding across timestep dimension.
        Maxpool in timestep dimension.
    """
    main_input = Input(shape=(_INPUT_LENGTH,), dtype='int32', name='main_input')
    x = main_input

    x = Embedding(output_dim=embedding_dim,
                  input_dim=_NUM_TOKENS,
                  input_length=_INPUT_LENGTH)(x)
    # output: (batch, input_length, output_dim)

    feature_maps = []
    for wlength in window_lengths:
        fmap = Conv1D(nb_filters,
                      wlength,
                      padding='same',
                      activation='relu',
                      kernel_regularizer=l2(l2_reg))(x)
        # (None, input_length, 8)
        if conv_dropout > 0.0:
            fmap = Dropout(conv_dropout)(fmap)
        feature_maps.append(fmap)

    x = Concatenate(axis=-1)(feature_maps)
    # (None, input_length, len(window_lengths)*8)

    x = GlobalMaxPooling1D()(x) # Pools along timestep dim
    # (None, len(window_lengths)*8)


    if type(nb_dense_units) is not list: # List of 1, if not a list
        nb_dense_units = [nb_dense_units]

    for nb_units in nb_dense_units:
        x = Dense(nb_units,
                  kernel_regularizer=l2(l2_reg))(x)
        x = Activation('relu')(x)
        if dense_dropout > 0.0:
            x = Dropout(dense_dropout)(x)

    # Apply final dense layer to classify into 11 authors
    # Keras categorical_crossentropy expects softmax
    x = Dense(_NUM_AUTHORS, name='classification_output')(x)
    x = Activation('softmax')(x)

    printl(main_input, 'main_input')
    printl(x, 'output')
    model = Model(inputs=main_input, outputs=x)
    model.name = "conv_model"

    return model


@testable
def vanilla_LSTM_model():
    """ Standard LSTM model.
        Dense layers after output for more power, then classification.
    """
    return __LSTM_model(nb_lstm_units=256,
                        nb_dense_units=128,
                        lstm_dropout=0.2,
                        dense_dropout=0.2,
                        model_name="vanilla_LSTM_model")

@testable
def stacked_2_LSTM_model():
    """ Two LSTMs stacked on top of one another, but learning less params each.
    """
    return __LSTM_model(nb_lstm_units=[128, 128],
                        nb_dense_units=64,
                        lstm_dropout=0.2,
                        dense_dropout=0.2,
                        return_sequences=True,
                        model_name="stacked_2_LSTM_model")

@testable
def stacked_3_LSTM_model():
    """ Two LSTMs stacked on top of one another, but learning less params each.
    """
    return __LSTM_model(nb_lstm_units=[128, 128, 128],
                        nb_dense_units=64,
                        lstm_dropout=0.2,
                        dense_dropout=0.2,
                        return_sequences=True,
                        model_name="stacked_3_LSTM_model")

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
                 go_backwards=False,
                 # Miscellaneous
                 model_name=None):
    # input: (batch, input_length, num_classes) tensor. one-hots already expanded
    main_input = Input(shape=(_INPUT_LENGTH, _NUM_TOKENS,), dtype='float32', name='main_input')
    x = main_input

    # If we are doing stacked lstm, need return_sequences=True
    assert (type(nb_lstm_units) is list) == return_sequences

    if type(nb_lstm_units) is not list: # List of 1, if not a list
        nb_lstm_units = [nb_lstm_units]

    for indx, nb_units in enumerate(nb_lstm_units):
        print("Adding a LSTM layer with %d units, dropout %.03f."
              % (nb_units, lstm_dropout))

        # The last LSTM layer must have return_sequences=False.
        if indx == len(nb_lstm_units)-1:
            return_sequences = False

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
    # Keras categorical_crossentropy expects softmax
    x = Dense(_NUM_AUTHORS, name='classification_output')(x)
    x = Activation('softmax')(x)

    printl(main_input, 'main_input')
    printl(x, 'output')
    model = Model(inputs=main_input, outputs=x)

    # Name the model so we can save its name later
    if model_name is not None:
        model.name = model_name
    else:
        # e.g. "LSTM_128_128"
        model.name = "LSTM_model_" + "_".join(map(str, nb_lstm_units))

    return model


if __name__ == "__main__":
    print()

    for model_fn in testable_models:
        print()
        print("Found model: %s." % model_fn.__name__)
        model = model_fn()
        model.summary()
