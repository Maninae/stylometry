import numpy as np
from scripts.read_data import load_data

class Config(object):
    batch_size = 32;
    epochs = 10;

config = Config()

class ModelOperations(object):
    self.config = Config()

    def train(self, model):
        X_train, Y_train = load_data('train')
        X_val, Y_val = load_data('val')
        model.compile(loss='categorial_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x=X_train,
                  y=Y_train,
                  batch_size=config.batch_size,
                  epochs=config.epochs,
                  verbose=1,
                  validation_data=(X_val, Y_val),
                  shuffle=True)


    def test(self, model):
        X_test, Y_test = load_data('test')
        model.evaluate(x=X_test,
                       y=Y_test,
                       batch_size=config.batch_size,
                       verbose=1)
