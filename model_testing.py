"""
File: model_testing.py
Author: Jacob Komissar
Date: 2017-10-07

File for testing various models.
"""
from functools import partial
from itertools import takewhile, dropwhile, count
import operator as op
from typing import Callable, List, Union, Iterable

from data import Data
from models import Layer, Network
from extra import repeat_n

COUNT=28*28

def get_data(seed=123134):
    "Get data to test a network on."
    return Data("images.npy", "labels.npy", seed=seed) # Test data


"""
Functions for generating neural networks
"""

def narrowing_sizes(f: Callable[[int], int], maximum=COUNT, minimum=10) -> List[int]:
    """Get the sizes for layers according to the given function.

    The minimum is not inclusive.
    """
    return takewhile(partial(op.lt, minimum),
                     dropwhile(partial(op.lt, maximum),
                               (int(f(i)) for i in count(0))))
    #return takewhile(partial(op.lt, minimum), (int(f(i)) for i in count(0)))


def narrowing_by(f: Callable[[int], int]) -> List[Layer]:
    "List of layers with sizes determined by the given function, down to 10."

    def make_default_layer(size, more={}):
        return Layer(size, 'RandomUniform', 'selu', **more)

    sizes = list(narrowing_sizes(f))
    if len(sizes) == 0:
        raise Exception("Not enough layers")
    elif len(sizes) == 1:
        layers = [Layer(sizes[0], 'RandomUniform', 'tanh')]
    else:
        layers = [make_default_layer(sizes[0], dict(input_shape=(28*28,)))]
        layers += [make_default_layer(size) for size in sizes[1:-1]]
        layers.append(Layer(sizes[-1], 'RandomUniform', 'tanh'))
    return layers


"""
DSL for building networks
"""

import keras.initializers as init
ru = 'RandomUniform'
rn = 'RandomNormal'
tn = 'TruncatedNormal'
vs = 'VarianceScaling' # keras.initializers.VarianceScaling(distribution='uniform') is also good
og = 'Orthogonal'

def network(*layers: List[Union[Layer, Iterable[Layer]]]) -> Network:
    actual = []
    for layer in layers:
        if isinstance(layer, Layer.cls):
            actual.append(layer)
        else:
            actual.extend(layer)
    return Network(*actual)

def layer(act='selu', dist=ru, n=COUNT) -> Layer:
    return Layer(n, dist, act)

def relu(n=COUNT, dist=ru): return Layer(n, dist, 'relu')
def selu(n=COUNT, dist=ru): return Layer(n, dist, 'selu')
def tanh(n=COUNT, dist=ru): return Layer(n, dist, 'tanh')

repeat = repeat_n



"""
Networks
"""

SELUS = 3
EPOCHS = 10
BATCH_SIZE = 64

# Compare distributions
wides = {
    'ru': network(
        repeat(SELUS, selu(ru)),
        repeat(1, tanh(ru))
    ).epochs(EPOCHS).batch_size(BATCH_SIZE),
    'rn': network(
        repeat(SELUS, selu(rn)),
        repeat(1, tanh(rn))
    ).epochs(EPOCHS).batch_size(BATCH_SIZE),

    'tn': network(
        repeat(SELUS, selu(tn)),
        repeat(1, tanh(tn))
    ).epochs(EPOCHS).batch_size(BATCH_SIZE),
    'og': network(
        repeat(SELUS, selu(og)),
        repeat(1, tanh(og))
    ).epochs(EPOCHS).batch_size(BATCH_SIZE),
}
# TruncatedNormal weights works very poorly with low batch sizes, but well with large sizes

vary_batches = [
    network(selu(), layer(tanh))
    , network(selu(), tanh()).epochs(8).batch_size(50)
    , network(selu(), tanh()).epochs(8).batch_size(64)
    , network(selu(), tanh()).epochs(8).batch_size(100)
    , network(selu(), tanh()).epochs(8).batch_size(200)
]


# Narrowing by eighths
narrowing = Network(*narrowing_by(lambda x: COUNT-(COUNT/8)*x))

narrowing2 = Network(*narrowing_by(lambda x: (10*(COUNT/10)**((10-x)/10))))


"""
BEST NETWORKS
"""

# This one takes almost no epochs to do very well.
fast = network(tanh(n=COUNT*5, dist=og)).epochs(2).batch_size(60)
# It often hits 90% accuracy after just one epoch.

def favorite(n: int) -> Network:
    "Make a network consisting of a number of selu layers, and a tanh layer."
    return network(
        repeat(n, selu()),
        tanh()
    ).epochs(8).batch_size(20)

# Final submission
ninety_four = favorite(4) # Hit 94 the first 3 times I ran it.

final = favorite(3) # Submitted


# 876543234 as a data seed gets excellent results from final (0.9382 on first epoch)
# Test on image 5483, the stupid-looking number 1
