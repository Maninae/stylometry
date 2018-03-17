from model.model import __testable_models as Models
from os.path import join
import pickle

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
TEST_DIR = 'data/test'


class Config(object):
    batch_size = 32
    epochs = 30


def load_data(split='train'):
    if split == 'train':
        d = TRAIN_DIR
    if split == 'val':
        d = VAL_DIR
    if split == 'test':
        d = TEST_DIR
    return (pickle.load(open(join(d, 'tokens.pkl'), 'rb')),
            pickle.load(open(join(d, 'authors.pkl'), 'rb')))


class ModelOperations(object):

    def __init__(self):
        self.config = Config()

    def train(self, model):
        X_train, Y_train = load_data('train')
        X_val, Y_val = load_data('val')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x=X_train,
                  y=Y_train,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epochs,
                  verbose=1,
                  validation_data=(X_val, Y_val),
                  shuffle=True)

    def test(self, model):
        X_test, Y_test = load_data('test')
        model.evaluate(x=X_test,
                       y=Y_test,
                       batch_size=self.config.batch_size,
                       verbose=1)


if __name__ == '__main__':
    ops = ModelOperations()
    print('Available models:')
    for i, model in enumerate(Models):
        print(i, model)
    print('Choose one (enter a number from 0 to %d):' % (len(Models) - 1))
    i = int(input())
    model = Models[i]()
    ops.train(model)
    ops.test(model)
