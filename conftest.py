from itertools import product

import keras_core as keras
import numpy as np
import pytest
import torch

import kornia
from testing.dtypes import DTYPES, load_dtype


@pytest.fixture()
def dtype(dtype_name) -> str:
    return load_dtype(dtype_name)


@pytest.fixture()
def channels_axis(channels_order):
    if channels_order == 'last':
        keras.backend.set_image_data_format('channels_last')
        return -1
    elif channels_order == 'first':
        keras.backend.set_image_data_format('channels_first')
        return -3


def pytest_generate_tests(metafunc):
    dtype_names = None
    channels_order_names = None

    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        if raw_value == 'all':
            dtype_names = list(DTYPES)
        else:
            dtype_names = raw_value.split(',')

    if 'channels_order' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--channels_order')
        if raw_value == 'all':
            channels_order_names = ['first', 'last']
        else:
            channels_order_names = raw_value.split(',')

    if dtype_names is not None and channels_order_names is not None:
        params = list(product(dtype_names, channels_order_names))
        metafunc.parametrize('dtype,channels_order', params)

    elif dtype_names is not None:
        metafunc.parametrize('dtype', dtype_names)

    elif channels_order_names is not None:
        metafunc.parametrize('channels_order', channels_order_names)


def pytest_addoption(parser):
    parser.addoption('--dtype', action="store", default="float32")
    parser.addoption('--channels_order', action="store", default="all")


@pytest.fixture(autouse=True)
def add_doctest_deps(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["torch"] = torch
    doctest_namespace["kornia"] = kornia
