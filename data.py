"""
File: data.py

Data structores and input.
"""
import random
from typing import NamedTuple, List
from warnings import warn

import numpy as np

class Image(NamedTuple):
    "Represents an image and its value, with a unique identifier."
    data: np.ndarray # The image.
    of: int          # The number of which this is an image.
    id: int = None   # The index of this number in the input files, or None.
    def __str__(self):
        return f"Image(of={self.of}, id={self.id}\n{image2str(self.data)}\n)"

    def oneHot(self) -> np.ndarray:
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
        onepercent = len(images) // 100
        r = round(onepercent * training)
        v = round(onepercent * validation)
        # s = round(onepercent * test)

        Data._randomizer.seed(Data._default_seed if seed is None else seed)
        random.shuffle(images)

        self._training = images[:r]
        self._validation = images[r:r+v]
        self._test = images[r+v:]

    @property
    def training(self): return self._training
    @property
    def validation(self): return self._validation
    @property
    def test(self): return self._test

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
