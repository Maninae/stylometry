import keras.backend as K
from keras.callbacks import ModelCheckpoint

# CONSTANTS
_NUM_AUTHORS = 11
_NUM_TOKENS = 12
_INPUT_LENGTH = 250


# Dictionary for weighting the loss on each author's data.
# Authors with fewer data samples will have a greater loss, to
#   avoid models biasing themselves toward higher freq authors.
author_counts = {
    0: 1910.,
    1: 2765.,
    2: 1704.,
    3: 1657.,
    4: 2260.,
    5: 4104.,
    6: 3335.,
    7: 6397.,
    8: 5169.,
    9: 12571.,
    10: 1816.
}
class_weight_dict = {k: max(author_counts.values()) / author_counts[k] for k in
                     author_counts}

# EDIT: Unlikely to swap out any LSTM activations for this.
#   Can swish activate dense layers, but not deep enough.


def swish(x):
    return x * K.sigmoid(x)


custom_objects_dict = {
    'swish': swish
}


def get_checkpointer(model_name):
    # Save everything, just in case
    filepath = "weights/%s/%s-ep{epoch:02d}-loss={loss:.4f}-vloss={val_loss:.4f}-tacc={acc:.3f}-vacc={val_acc:.3f}.h5" % (model_name, model_name)
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=1,
                                   save_best_only=False)

    return checkpointer
