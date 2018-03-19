import argparse
import numpy as np
import pickle
from datetime import datetime

from keras.models import load_model

from model import stylometry_models as sm

from utils.dataflow import load_data, preprocess_input

valid_model_names = [fn.__name__ for fn in sm.testable_models]

models_requiring_preprocess = [
    "vanilla_LSTM_model",
    "stacked_2_LSTM_model",
    "stacked_3_LSTM_model"
]


def test(self, model, preprocess=True):
    X_test, Y_test = load_data('test')

    if preprocess:  # Preprocessing turns ints into one-hot vectors
        X_test = preprocess_input(X_test)

    eval_output = model.evaluate(x=X_test,
                                 y=Y_test,
                                 batch_size=128,
                                 verbose=1)

    labeled_eval_output = zip(model.metrics_names, eval_output)
    # ret a list of [('loss', scalar), ('accuracy', <scalar>), ...]
    return labeled_eval_output


##############################################


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", dest='stored_model_path',
        help="load an existing model")

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
python3 test.py --load_path conv_model
python3 test.py --load_path vanilla_LSTM_model
python3 test.py --load_path stacked_2_LSTM_model
python3 test.py --load_path stacked_3_LSTM_model
"""

if __name__ == "__main__":
    args = parse_arguments_from_command()
    print("[db-testing] We got args: %s" % str(args))
    model_name = args.model_name
    stored_model_path = args.stored_model_path

    model = load_model(stored_model_path)
    assert model is not None

    test(model, preprocess=model_name in models_requiring_preprocess)
