import numpy as np
import pickle
from datetime import datetime

from model import stylometry_models as sm
from model.model_operations import ModelOperations
from model.util import get_checkpointer, class_weight_dict

from utils.dataflow import load_data, preprocess


valid_model_names = [fn.__name__ for fn in sm.testable_models]

models_requiring_preprocess = [
    "vanilla_LSTM_model",
    "stacked_2_LSTM_model",
    "stacked_3_LSTM_model"
]

# Default config settings for model training
class Config(object):
    batch_size = 32
    epochs = 30
    optimizer = 'adam'
    loss = 'categorial_crossentropy'
    metrics_list = ['accuracy']


##########################################

def train(self, model, preprocess=True):
    X_train, Y_train = load_data('train')
    X_val, Y_val = load_data('val')

    if preprocess: # Preprocessing turns ints into one-hot vectors
        X_train = preprocess(X_train)
        X_val = preprocess(X_val)

    model.compile(loss=self.config.loss,
                  optimizer=self.config.optimizer,
                  metrics=self.config.metrics_list)
    
    checkpointer = get_checkpointer(model.name)

    history = model.fit(x=X_train,
                        y=Y_train,
                        batch_size=self.config.batch_size,
                        epochs=self.config.epochs,
                        class_weight=class_weight_dict,
                        verbose=1,
                        validation_data=(X_val, Y_val),
                        callbacks=[checkpointer],
                        shuffle=True)
    # ret a dict of {'loss': [0.294, 0.290, ...], 'accuracy': [], ...}
    return history.history


def test(self, model, preprocess=True):
    X_test, Y_test = load_data('test')

    if preprocess: # Preprocessing turns ints into one-hot vectors
        X_test = preprocess(X_test)

    eval_output = model.evaluate(x=X_test,
                                 y=Y_test,
                                 batch_size=self.config.batch_size,
                                 verbose=1)
    
    labeled_eval_output = zip(model.metrics_names, eval_output)
    # ret a list of [('loss', scalar), ('accuracy', <scalar>), ...]
    return labeled_eval_output


##############################################


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
            help="Debug mode: load instead the coco-debug directory as data",
            action='store_true')
    parser.add_argument("--load_path", dest='stored_model_path',
            help="optional path argument, if we want to load an existing model")
    parser.add_argument("--new_model", dest='model_name',
            help="Type of model in model.py we want to run")

    def validate_args(args):
        """ Validations that the args are correct """
        assert args.model_name in valid_model_names
        assert not bool(args.load_path and args.model_name)

    args = parser.parse_args()
    validate_args(args)
    return args


"""
Usage:
python3 train.py --new_model conv_model
python3 train.py --new_model vanilla_LSTM_model
python3 train.py --new_model stacked_2_LSTM_model
python3 train.py --new_model stacked_3_LSTM_model
"""

if __name__ == "__main__":
    args = parse_arguments_from_command()    
    print("[db-training] We got args: %s" % str(args))
    model_name = args.model_name
    stored_model_path = args.stored_model_path

    skip_loading = True
    if stored_model_path is not None and not skip_loading:
        # TBD: Load an existing model
        model_fn = None
        model = None
    else:
        model_fn = getattr(sm, model_name)
        model = model_fn()

    assert model is not None

    config = Config()
    history = train(model, preprocess = model_name in models_requiring_preprocess)
    # Save the history to plot later
    with open('histories/%s_%s.pkl' % (model.name, datetime.now()), 'wb') as f:
        pickle.dump(history, f)

