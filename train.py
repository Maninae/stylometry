import argparse
import numpy as np
import pickle
from datetime import datetime

from keras.optimizers import Adam
from keras.models import load_model

from model import stylometry_models as sm
from model.util import get_checkpointer, class_weight_dict, get_lr_callback

from utils.dataflow import load_data, preprocess_input

valid_model_names = [fn.__name__ for fn in sm.testable_models]

models_requiring_preprocess = [
    "vanilla_LSTM_model",
    "stacked_2_LSTM_model"
]

# Default config settings for model training


class Config(object):
    batch_size = 128
    epochs = 30
    optimizer = Adam(lr=1e-4)
    loss = 'categorical_crossentropy'
    metrics_list = ['accuracy']


config = Config()

##########################################


def train(model, preprocess=True):
    X_train, Y_train = load_data('train')
    print(X_train.shape, Y_train.shape)
    X_val, Y_val = load_data('val')

    if preprocess:  # Preprocessing turns ints into one-hot vectors
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)

    print(X_train.shape)

    model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics_list)

    checkpointer = get_checkpointer(model.name)
    lr_reducer_on_plateau = get_lr_callback()

    history = model.fit(x=X_train,
                        y=Y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        class_weight=class_weight_dict,
                        verbose=1,
                        validation_data=(X_val, Y_val),
                        callbacks=[checkpointer, lr_reducer_on_plateau],
                        shuffle=True)
    # ret a dict of {'loss': [0.294, 0.290, ...], 'accuracy': [], ...}
    return history.history


def test(self, model, preprocess=True):
    X_test, Y_test = load_data('test')

    if preprocess:  # Preprocessing turns ints into one-hot vectors
        X_test = preprocess_input(X_test)

    eval_output = model.evaluate(x=X_test,
                                 y=Y_test,
                                 batch_size=config.batch_size,
                                 verbose=1)

    labeled_eval_output = zip(model.metrics_names, eval_output)
    # ret a list of [('loss', scalar), ('accuracy', <scalar>), ...]
    return labeled_eval_output


##############################################


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        help="Debug mode: load instead the coco-debug directory as data",
        action='store_true')
    parser.add_argument(
        "--load_path", dest='stored_model_path',
        help="optional path argument, if we want to load an existing model")
    parser.add_argument("--new_model", dest='model_name',
                        help="Type of model in model.py we want to run")
    parser.add_argument("--epochs", dest='epochs', type=int,
                        help="Num epochs to run for")
    parser.add_argument("--batch_size", type=int,
                        help="batch size during train/val")

    def validate_args(args):
        """ Validations that the args are correct """
        if args.model_name is not None:
            assert args.model_name in valid_model_names
        assert not bool(args.stored_model_path and args.model_name)

    args = parser.parse_args()
    validate_args(args)
    return args


"""
Usage:
python3 train.py --new_model conv_model
python3 train.py --new_model vanilla_LSTM_model
python3 train.py --new_model stacked_2_LSTM_model
"""

if __name__ == "__main__":
    args = parse_arguments_from_command()
    print("[db-training] We got args: %s" % str(args))
    model_name = args.model_name
    stored_model_path = args.stored_model_path

    skip_loading = False
    if stored_model_path is not None and not skip_loading:
        # TBD: Load an existing model
        model = load_model(stored_model_path)
        model_name = model.name # Because no argument populating model_name
    else:
        model_fn = getattr(sm, model_name)
        model = model_fn()

    assert model is not None

    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    history = train(model, preprocess=model_name in models_requiring_preprocess)
    # Save the history to plot later
    with open('histories/%s/%s_%s.pkl' % (model.name, model.name, datetime.now()), 'wb') as f:
        pickle.dump(history, f)
