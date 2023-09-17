import os

import keras_core as keras


def test_backend():
    assert os.environ["KERAS_BACKEND"] == keras.backend.backend()
