import numpy as np
from scripts.read_data import load_data
from model import model as stylometry_models


valid_model_names = [fn.__name__ for fn in stylometry_models.testable_models]


class Config(object):
    batch_size = 32
    epochs = 10
    optimizer = 'adam'
    loss = 'categorial_crossentropy'
    metrics_list = ['accuracy']

class ModelOperations(object):
    def __init__(self):
        self.config = Config()

    def train(self, model):
        X_train, Y_train = load_data('train')
        X_val, Y_val = load_data('val')
        model.compile(loss=self.config.loss,
                      optimizer=self.config.optimizer,
                      metrics=self.config.metrics_list)
        
        history = model.fit(x=X_train,
                            y=Y_train,
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            verbose=1,
                            validation_data=(X_val, Y_val),
                            shuffle=True)
        # ret a dict of {'loss': [0.294, 0.290, ...], 'accuracy': [], ...}
        return history.history


    def test(self, model):
        X_test, Y_test = load_data('test')
        eval_output = model.evaluate(x=X_test,
                                     y=Y_test,
                                     batch_size=self.config.batch_size,
                                     verbose=1)
        
        labeled_eval_output = zip(model.metrics_names, eval_output)
        # ret a list of [('loss', scalar), ('accuracy', <scalar>), ...]
        return labeled_eval_output


def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
            help="Debug mode: load instead the coco-debug directory as data",
            action='store_true')
    parser.add_argument("--load_path", dest='stored_model_path',
            help="optional path argument, if we want to load an existing model")
    parser.add_argument("--model", dest='model_name',
            help="Type of model in model.py we want to run")

    def validate_args(args):
        """ Validations that the args are correct """
        assert args.model_name in valid_model_names

    args = parser.parse_args()
    validate_args(args)
    return args

if __name__ == "__main__":
    args = parse_arguments_from_command()    
    print("[db-training] We got args: %s" % str(args))

    skip_loading = True
    if args.stored_model_path is not None and not skip_loading:
        # TBD: Load an existing model
        model_fn = None
        model = None
    else:
        model_fn = getattr(stylometry_models, args.model_name)
        model = model_fn()

    assert model is not None

    model_ops = ModelOperations()
    model_ops.train(model)

