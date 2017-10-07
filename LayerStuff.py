import csv
from typing import List, Iterable, Dict, NamedTuple

from keras.initializers import *
from keras.activations import *
from keras.layers import Dense
from keras.models import Sequential
import itertools
from data import Data


EPOCHS: int = 10
BATCH_SIZE: int = 512

denseLayerConfig = {
    'activation': ('relu', 'selu', 'tanh'),
    'kernel_initializer': ('he_normal', 'RandomNormal', 'RandomUniform')
}
LayerConfig = NamedTuple('LayerConfig', [(field_name, str) for field_name in denseLayerConfig.keys()])


def all_combos(in_dict: Dict[object, Iterable]):
    '''
    returns all dicts in which for each key, the key remains the same but the value is an element chosen
    from the original value which is an iterable
    :param in_dict: 
    :return: 
    '''
    for tuple in itertools.product(*in_dict.values()):
        yield dict(zip(denseLayerConfig.keys(), tuple))


def exhaustive_interpolate_vectors(a: float, b: float, dim: int) -> List[List[int]]:
    def interpolate_exp(a: float, b: float, x_list: np.ndarray) -> np.ndarray:
        '''
        Interpolate values between the points (0,a) and (1,b) using the function a*(b/a)^x
        :param a: 
        :param b: 
        :param x_list: List of values of x for which the interpolation is done
        :return: List corresponding to interpolated values
        '''
        return a * np.power(b / a, x_list)

    interpolated_vals = [int(np.ceil(val)) for val in
                         interpolate_exp(a, b, np.linspace(-0.2, 1.2, 11, endpoint=True)).tolist()
                         if val >= 0]  # type: List[int]
    return list(itertools.product(interpolated_vals, repeat=dim))


def test_shapes_suite(data: Data, num_layers: int, layer_configs: List[List[LayerConfig]]):
    tests = []  # type: List[TestRecord]
    results = []  # type: List[RelevantResults]
    for shape in exhaustive_interpolate_vectors(10, 28*28, num_layers):
        for layer_list in layer_configs:
            model = Sequential()
            for i, (units, layer_config) in enumerate(zip(shape, layer_list)):
                if i==0:
                    model.add(make_dense_layer_from_config(units, layer_config, input_shape=(28*28,)))
                else:
                    model.add(make_dense_layer_from_config(units, layer_config))
            model.add(Dense(10, kernel_initializer='he_normal', activation='softmax'))
            # Compile Model
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

            history, confusion_matrix = test_model(model, data)
            test_rec = TestRecord(shape, layer_list, history, confusion_matrix)
            tests.append(test_rec)
            rel_res = RelevantResults(
                test_rec.config_text,
                test_rec.history.history['val_acc'][-1],
                test_rec.history.history['acc'][-1],
                shape)
            results.append(rel_res)
            print(rel_res)

    with open('test_results.csv', 'a+') as fp:
        write_named_tuples(list(itertools.chain(results)), fp)


def test_shapes(data: Data, layer_configs: List[List[LayerConfig]], shapes: List[List[int]]):
    tests = []  # type: List[TestRecord]
    results = []  # type: List[RelevantResults]
    for shape in shapes:
        for layer_list in layer_configs:
            print(f'testing {layer_list}')
            model = Sequential()
            for i, (units, layer_config) in enumerate(zip(shape, layer_list)):
                if i==0:
                    model.add(make_dense_layer_from_config(units, layer_config, input_shape=(28*28,)))
                else:
                    model.add(make_dense_layer_from_config(units, layer_config))
            model.add(Dense(10, kernel_initializer='he_normal', activation='softmax'))
            # Compile Model
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

            history, confusion_matrix = test_model(model, data)
            test_rec = TestRecord(shape, layer_list, history, confusion_matrix)
            tests.append(test_rec)
            rel_res = RelevantResults(
                test_rec.config_text,
                test_rec.history.history['val_acc'][-1],
                test_rec.history.history['acc'][-1],
                shape)
            results.append(rel_res)
            print(rel_res)

    with open('test_results.csv', 'a+') as fp:
        write_named_tuples(list(itertools.chain(results)), fp)


def make_confusion_matrix(predict_y: np.ndarray, real_y: np.ndarray):
    class PredictPair(NamedTuple):
        realY: int
        predictedY: int

    confusion_dict = {}  # type: Dict[PredictPair, int]
    for predict_y_entry, real_y_entry in zip(predict_y, real_y):
        pair = PredictPair(real_y_entry, predict_y_entry)
        confusion_dict[pair] = confusion_dict.setdefault(pair, 0) + 1

    return confusion_dict


def make_dense_layer_from_config(units, LayerConfig, **kwargs):
    return Dense(units, **{**kwargs, **LayerConfig._asdict()})


class RelevantResults(NamedTuple):
    config_text: str
    val_acc: float
    acc: float
    shape: List[int]


class TestRecord(NamedTuple):
    shape: List[int]  # The shape of the layers
    layer_list_configs: List[LayerConfig]
    history: dict
    confusion_matrix: dict

    @property
    def config_text(self) -> str:
        return '--'.join(['|'.join(layer_config._asdict().values()) for layer_config in self.layer_list_configs])


def test_model(model: Sequential, data: Data):
    # Train Model
    history = model.fit(
        np.vstack([img.data for img in data.training]),
        np.vstack([img.one_hot() for img in data.training]),
        validation_data=(
            np.vstack([img.data for img in data.validation]),
            np.vstack([img.one_hot() for img in data.validation])),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE)

    predicted_y = model.predict(np.vstack([img.data for img in data.test]), batch_size=BATCH_SIZE)
    confusion_matrix = make_confusion_matrix(np.argmax(predicted_y, axis=1),
                                             np.hstack([img.of for img in data.test]))

    return history, confusion_matrix


def write_named_tuples(tuples: List[NamedTuple], fp):
    wr = csv.DictWriter(fp, delimiter=',', fieldnames=list(tuples[0]._asdict().keys()))
    wr.writeheader()
    for tuple in tuples:
        wr.writerow(tuple._asdict())


def test_layer_configs(shape_list: List[List[int]], data: Data, num_repeats=1):
    def get_models_layer_configs(num_layers: int) -> List[List[LayerConfig]]:

        return list(itertools.product(
            [LayerConfig(**layer_config) for layer_config in all_combos(denseLayerConfig)],
            repeat=num_layers))

    tests = []  # List[TestRecord]
    results = {}  # type: Dict[tuple, List[RelevantResults]]
    for shape in shape_list:
        for layer_list in get_models_layer_configs(len(shape)):
            model = Sequential()
            for i, (layer_config, num_nodes_in_layer) in enumerate(zip(layer_list, shape)):
                if i == 0:
                    model.add(make_dense_layer_from_config(num_nodes_in_layer, layer_config, input_shape=(28*28,)))
                else:
                    # model.add(Dense(num_nodes_in_layer, **layer_config._asdict()))
                    model.add(make_dense_layer_from_config(num_nodes_in_layer, layer_config))
            model.add(Dense(10, kernel_initializer='he_normal', activation='softmax'))
            # Compile Model
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

            for _ in range(num_repeats):
                history, confusion_matrix = test_model(model, data)
                test_rec = TestRecord(shape, layer_list, history, confusion_matrix)
                tests.append(test_rec)
                rel_res = RelevantResults(
                    test_rec.config_text,
                    test_rec.history.history['val_acc'][-1],
                    test_rec.history.history['acc'][-1],
                    shape)
                results.setdefault(tuple(shape), []).append(rel_res)
                print(rel_res)

    with open('test_results.csv', 'a+') as fp:
        write_named_tuples(list(itertools.chain(results.values())), fp)
