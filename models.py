from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from data import Data

EPOCHS: int = 10
BATCH_SIZE: int = 512


class Model(object):
    def __init__(self, *layers):
        self.model = Sequential(*layers) # declare model
        self.model.add(Dense(10, kernel_initializer='he_normal', activation='softmax'))
        self.model.compile(optimizer='sgd',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def test(self, data: Data):
        history = self.model.fit(
            np.vstack([img.data for img in data.training]),
            np.vstack([img.one_hot() for img in data.training]),
            validation_data=(
                np.vstack([img.data for img in data.validation]),
                np.vstack([img.one_hot() for img in data.validation])),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE)
        return history


test = Model(
    [Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal', activation='relu')]

)

    # Train Model
