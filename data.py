"""
File: data.py

Data structores and input.
"""
from collections import namedtuple
from functools import reduce
import itertools
from operator import attrgetter as getter
import random
from typing import NamedTuple, List

import numpy as np

MiniData = namedtuple("MiniData", ("training", "validation", "test"))

def groupby(iterable, key=None):
    """Version of itertools.groupby that sorts by the key first
    """
    return itertools.groupby(sorted(iterable, key=key), key=key)

class Image(NamedTuple):
    "Represents an image and its value, with a unique identifier."
    data: np.ndarray # The image.
    of: int          # The number of which this is an image.
    id: int = None   # The index of this number in the input files, or None.
    def __str__(self):
        return f"Image(of {self.of}, id={self.id}\n{image2str(self.data)}\n)"

    def one_hot(self) -> np.ndarray:
        """Get the encoded number as a vector.

        For example, Image(..., 5) would become [0,0,0,0,0,1,0,0,0,0].
        """
        array = np.zeros(10, dtype=int)
        array[self.of] = 1
        array.flags.writeable = False
        return array

class Data(object):
    """Represents data for use in a neural network."""

    _default_seed = 20171002
    _randomizer = random.Random(20171002)

    def __init__(self, imagefile, labelfile,
                 *, training=60, validation=15, test=25, seed=None):
        """Use the data in the given files to create a data set.

        The training, validation, and test sets' sizes are given as integer
        percentages. If they do not evenly divide the data, remaining data
        will be added to the test set.
        """
        if training + validation + test != 100:
            raise ValueError("Sum of data sets is not 100")
        images = load_files(imagefile, labelfile)


        bynum = {i: tuple(imgs) for i, imgs in groupby(images, key=getter('of'))}

        train = []
        valid = []
        test = []
        testids = []
        stats = {}
        for num, nimages in bynum.items():
            r = round(len(nimages) * training // 100)
            v = round(len(nimages) * validation // 100)
            train += nimages[:r]
            valid += nimages[r:r+v]
            test  += nimages[r+v:]
            stats[num] = MiniData(r, v, len(nimages)-r-v)
        # Now train, valid, and test are the data, ordered by number

        # This is the best time to convert the lists into dicts, if you want to.

        """
         I want to support these access methods:
        Get by number : int -> ImmutableCollection[Image]
        -- Get by set    : (set ->) ImmutableCollection[Image]
        Get set as dict: set -> { int: Image }
        Get by both   : set -> int -> ImmutableCollection[Image]
        Get as dict : -> { set: { int: Image } }
        -- Get as set
        """

        data = train + valid + test
        Data._randomizer.seed(Data._default_seed if seed is None else seed)
        Data._randomizer.shuffle(train)
        Data._randomizer.shuffle(valid)
        Data._randomizer.shuffle(test)
        Data._randomizer.shuffle(data)
        self._training = tuple(train)
        self._validation = tuple(valid)
        self._test = tuple(test)
        self._alldata = tuple(data)

        self.stats = stats

    @property
    def training(self) -> List[Image]: return self._training
    @property
    def validation(self) -> List[Image]: return self._validation
    @property
    def test(self) -> List[Image]: return self._test
    @property
    def alldata(self): return self._alldata

    def by_id(self, id: int):
        return next(filter(lambda x: x.id == id, self._alldata))

    def training_of(self, n: int):
        "Get all training images of the given number."
        return tuple(x for x in self.training if x.of == n)

    def validation_of(self, n: int):
        "Get all validation images of the given number."
        return tuple(x for x in self.validation if x.of == n)

    def test_of(self, n: int):
        "Get all test images of the given number."
        return tuple(x for x in self.test if x.of == n)

    def images_of(self, n: int):
        "Get all images of the given number."
        return tuple(x for x in self.alldata if x.of == n)

    def print_stats(self):
        print('   TRN VAL TST', *(
            f"{num}: {vals.training:>3} {vals.validation:>3} {vals.test:>3}"
            for num, vals in self.stats.items()), sep='\n')

def image2str(arr: np.ndarray, char='X', *, columns=28) -> str:
    """Convert image data from an array to a string.

    Data should be given as a 1-d array.

    Nonzero characters will be rendered as the given character, and zeros will
    be rendered as spaces.
    """
    if isinstance(arr, np.ndarray):
        dims = len(arr.shape)
    else:
        dims, a = 0, arr
        while True:
            try:
                a = a[0]
            except TypeError:
                break
            dims += 1
    if dims == 1:
        arr = [arr[i:i+columns] for i in range(0, len(arr), columns)]
        # for i in arr: print(' '.join(f'{x:>3}' for x in i)) # to print nicely as numbers
    elif dims != 2:
        raise ValueError("Can not convert many-dimensional array to string")
    return '\n'.join(''.join(char if i else ' ' for i in row) for row in arr)


def load_files(imagefile, labelfile) -> List[Image]:
    "Read an image file and a label file into a list of Images."
    result = []
    images = np.load(imagefile)
    labels = np.load(labelfile)
    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.flatten()
        image.flags.writeable = False
        result.append(Image(image.flatten(), label, i))
        i += 1
    return result
