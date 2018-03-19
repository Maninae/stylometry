import argparse
import numpy as np
import pickle
from datetime import datetime
from keras.models import load_model
from model import stylometry_models as sm
from sklearn.metrics import confusion_matrix
from utils.dataflow import load_data, preprocess_input

valid_model_names = [fn.__name__ for fn in sm.testable_models]

models_requiring_preprocess = [
    "vanilla_LSTM_model",
    "stacked_2_LSTM_model",
    "stacked_3_LSTM_model"
]


def test(model, preprocess=True):
    X_test, Y_test = load_data('test')

    if preprocess:  # Preprocessing turns ints into one-hot vectors
        X_test = preprocess_input(X_test)

    eval_output = model.evaluate(x=X_test,
                                 y=Y_test,
                                 batch_size=128,
                                 verbose=1)
    predictions = model.predict(x=X_test,
                                batch_size=128,
                                verbose=1)
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(y_true, y_pred)
    labeled_eval_output = zip(model.metrics_names + ['confusion'], eval_output + [confusion])
    # ret a list of [('loss', scalar), ('accuracy', <scalar>), ...]
    return labeled_eval_output


##############################################


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", dest='stored_model_path',
        help="load an existing model")
    args = parser.parse_args()
    return args


"""
Usage:
python3 test.py --load_path [path]
"""

if __name__ == "__main__":
    args = parse_arguments_from_command()
    print("[db-testing] We got args: %s" % str(args))
    stored_model_path = args.stored_model_path

    model = load_model(stored_model_path)
    model_name = model.name
    assert model is not None

    output = test(model, preprocess=model_name in models_requiring_preprocess)
    print(*output)
    with open('evals/%s/%s_%s.pkl' % (model.name, model.name, datetime.now()), 'wb') as f:
        pickle.dump(output, f)
