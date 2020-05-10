import pytest


def pytest_generate_tests(metafunc):
    if 'device_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--device')
        metafunc.parametrize('device_name', raw_value.split(','))
    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        metafunc.parametrize('dtype_name', raw_value.split(','))


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--dtype', action="store", default="float32")
