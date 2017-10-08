#!/usr/bin/env python3
from itertools import product
from operator import attrgetter, itemgetter
from random import random
from typing import NamedTuple, List

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from LayerStuff import test_layer_configs, test_shapes_suite, test_shapes, LayerConfig

from data import Data
from model_testing import final

"""
CONSTANTS
"""

IMAGE_FILE: str = "images.npy"
LABEL_FILE: str = "labels.npy"

# Model Template

# This function might not work.
# 'fails' is right, but 'fail_ids' might not be.
def failures(model, data):
    "Get the images from the data's test set that are not correctly predicted by the model."
    # num_fails = sum(matrix[c1, c2] for c1, c2 in product(range(10), range(10)) if c1 != c2)

    # tuple of ((actual, predicted), image)
    tests = (model.testid(i) for i in data.test)
    fails = filter(lambda x: x[0] != x[1], tests)
    fail_ids = map(itemgetter(0), fails)
    fail_images = map(lambda x: data.by_id(x), fail_ids)

    return fails, fail_images


def main():

    data = Data(IMAGE_FILE, LABEL_FILE, seed=random()) # Test data, random seed

    # # test_layer_configs([[10, 10]], data, num_repeats=3)
    # # test_shapes_suite(data, 2, [[LayerConfig('tanh', 'RandomNormal')], [LayerConfig('tanh', 'RandomNormal')]])
    # test_shapes(
    #     data,
    #     [
    #         [LayerConfig('relu', 'RandomUniform'), LayerConfig('tanh', 'RandomUniform'), LayerConfig('tanh', 'RandomUniform')]
    #     ],
    #     [[1000, 1000, 1000]]
    # )

    model = final.get_model()
    history = model.train(data)
    matrix = model.run(matrix)


if __name__ == '__main__':
    main()

