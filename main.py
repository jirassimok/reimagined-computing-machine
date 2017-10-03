#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

from data import Data

"""
CONSTANTS
"""
EPOCHS: int = 10
BATCH_SIZE: int = 512

IMAGE_FILE: str = "images.npy"
LABEL_FILE: str = "labels.npy"


# Model Template

def main():
    data = Data(IMAGE_FILE, LABEL_FILE) # Test data

    model = Sequential() # declare model
    model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))
    #
    #
    #
    # Fill in Model Here
    #
    #
    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))


    # Compile Model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)


    # Report Results

    print(history.history)
    model.predict()


if __name__ == '__main__':
    main()

