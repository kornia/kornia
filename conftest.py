import pytest


def pytest_generate_tests(metafunc):
    if 'device_type' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--typetest')
        metafunc.parametrize('device_type', raw_value.split(','))
    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtypetest')
        metafunc.parametrize('dtype_name', raw_value.split(','))


def pytest_addoption(parser):
    parser.addoption('--typetest', action="store", default="cpu")
    parser.addoption('--dtypetest', action="store", default="float32")
