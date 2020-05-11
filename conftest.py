from typing import Dict

import torch
import pytest


def get_test_devices() -> Dict[str, torch.device]:
    """Creates a dictionary with the devices to test the source code.
    CUDA devices will be test only in case the current hardware supports it.

    Return:
        dict(str, torch.device): list with devices names.
    """
    devices: Dict[str, torch.device] = {}
    devices["cpu"] = torch.device("cpu")
    if torch.cuda.is_available():
        devices["cuda"] = torch.device("cuda:0")
    return devices


def get_test_dtypes() -> Dict[str, torch.dtype]:
    """Creates a dictionary with the dtypes the source code.

    Return:
        dict(str, torch.dtype): list with dtype names.
    """
    dtypes: Dict[str, torch.dtype] = {}
    dtypes["float16"] = torch.float16
    dtypes["float32"] = torch.float32
    dtypes["float64"] = torch.float64
    return dtypes


# setup the devices to test the source code

TEST_DEVICES: Dict[str, torch.device] = get_test_devices()
TEST_DTYPES: Dict[str, torch.dtype] = get_test_dtypes()


@pytest.fixture()
def device(device_name) -> torch.device:
    if device_name not in TEST_DEVICES:
        pytest.skip(f"Unsupported device type: {device_name}")
    return TEST_DEVICES[device_name]


@pytest.fixture()
def dtype(device, dtype_name) -> torch.dtype:
    if device.type == 'cpu' and dtype_name == 'float16':
        pytest.skip(f"Unsupported device cpu and dtype float16.")
    return TEST_DTYPES[dtype_name]


def pytest_generate_tests(metafunc):
    if 'device_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--device')
        if raw_value == 'all':
            device_names = list(TEST_DEVICES.keys())
        else:
            device_names = raw_value.split(',')
        metafunc.parametrize('device_name', device_names)
    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        if raw_value == 'all':
            dtype_names = list(TEST_DTYPES.keys())
        else:
            dtype_names = raw_value.split(',')
        metafunc.parametrize('dtype_name', dtype_names)


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--dtype', action="store", default="float32")
