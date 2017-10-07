#!/usr/bin/env python3
from typing import NamedTuple, List

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from LayerStuff import test_layer_configs, test_shapes_suite, test_shapes, LayerConfig


from data import Data

"""
CONSTANTS
"""

IMAGE_FILE: str = "images.npy"
LABEL_FILE: str = "labels.npy"

# Model Template


def main():

    data = Data(IMAGE_FILE, LABEL_FILE) # Test data

    # test_layer_configs([[10, 10]], data, num_repeats=3)
    # test_shapes_suite(data, 2, [[LayerConfig('tanh', 'RandomNormal')], [LayerConfig('tanh', 'RandomNormal')]])
    test_shapes(
        data,
        [
            [LayerConfig('relu', 'RandomUniform'), LayerConfig('tanh', 'RandomUniform'), LayerConfig('tanh', 'RandomUniform')]
        ],
        [[500, 500, 500]]
    )

if __name__ == '__main__':
    main()

