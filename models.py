"""
File: models.py
Date: 2017-10-07

Classes to represent models.
"""
from typing import NamedTuple, Tuple, List, Dict, Any, Optional

from keras.layers import Dense
from keras.models import Sequential
import numpy as np

from data import Data, Image
from extra import namedtuple_varargs

EPOCHS: int = 10
BATCH_SIZE: int = 64
COUNT: int = 28 * 28



"""
Utility functions
"""

def make_confusion_matrix(predicted_values, real_values):
    confusion_dict = {}  # type: Dict[PredictPair, int]
    matrix = np.zeros((10, 10), dtype=int)
    for predicted, real in zip(predicted_values, real_values):
        matrix[real, predicted] += 1
    return matrix


"""
Objects representing neural networks and compnents
"""

Layer = 'Layer' # For forward references
@namedtuple_varargs('additional')
class Layer(NamedTuple):
    "Represents a dense layer of a neural network."
    units: int
    kernel_initializer: str
    activation: str
    input_shape: Tuple[int] = None   # Do not set manually. Call as_first() instead.
    additional: Dict[str, Any] = {}

    def to_dense(self) -> Dense:
        "Convert to Dense object"
        kwargs = dict(self.additional)
        if self.input_shape is not None:
            kwargs['input_shape'] = self.input_shape
        return Dense(self.units,
                     kernel_initializer=self.kernel_initializer,
                     activation=self.activation,
                     **kwargs)

    def as_first(self, *, input_shape=(COUNT,)) -> Layer:
        "Copy this layer, with an input shape added."
        return self.__class__(self.units,
                              self.kernel_initializer,
                              self.activation,
                              input_shape,
                              self.additional)


class Model(object):
    "Represents a model, with convenient training and predicting methods."

    def __init__(self, model, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.model = model
        self.histories = []
        self.epochs = epochs
        self.batch_size = batch_size

    def history(self):
        "Get last training history."
        return self.histories[-1]

    def train(self, data: Data) -> 'Model':
        "Train this model on the given data, returning this model."
        history = self.model.fit(
            np.vstack([img.data for img in data.training]),
            np.vstack([img.one_hot() for img in data.training]),
            validation_data=(
                np.vstack([img.data for img in data.validation]),
                np.vstack([img.one_hot() for img in data.validation])),
            epochs=self.epochs,
            batch_size=self.batch_size)

        self.histories.append(history)
        return history

    def run(self, data: Data):
        "Run this model on the given data, returning the confusion matrix."
        predicted = self.model.predict(np.array([img.data for img in data.test]),
                                       batch_size=self.batch_size)
        confusion = make_confusion_matrix(np.argmax(predicted, axis=1),
                                          np.array([img.of for img in data.test]))

        return confusion

    def test(self, image: Image):
        result = self.model.predict(np.array([image.data]))
        guess = np.argmax(result)
        return image.of, guess

class Network(object):
    "Represents the layers of a model."

    def __init__(self, *layers: List[Layer], epochs=EPOCHS, batch_size=BATCH_SIZE):
        self._epochs = epochs
        self._batch_size = batch_size

        if len(layers) == 0:
            self.base_layers = [Layer(10, 'he_normal', 'softmax').as_first()]
            self.layers = tuple(map(Layer.to_dense, self.base_layers))
            return

        base_layers  = [layers[0].as_first()]
        base_layers += layers[1:]
        base_layers.append(Layer(10, 'he_normal', 'softmax'))

        self.base_layers = tuple(base_layers)
        self.layers = tuple(map(Layer.to_dense, base_layers))

    def epochs(self, epochs: int):
        self._epochs = epochs
        return self

    def batch_size(self, size: int):
        self._batch_size = size
        return self

    def get_model(self) -> Model:
        "Get a model based on these layers."
        model = Sequential(self.layers) # declare model
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return Model(model, epochs=self._epochs, batch_size=self._batch_size)

    def test(self, data: Data):
        "Test a new model based on this network, returning (history, confusion_matrix)"
        model = self.get_model()
        history = model.train(data)
        matrix = model.run(data)
        return history, matrix
