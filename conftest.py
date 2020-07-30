from itertools import product
from typing import Dict

import pytest
import torch


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

# Combinations of device and dtype to be excluded from testing.
DEVICE_DTYPE_BLACKLIST = {('cpu', 'float16')}


@pytest.fixture()
def device(device_name) -> torch.device:
    if device_name not in TEST_DEVICES:
        pytest.skip(f"Unsupported device type: {device_name}")
    return TEST_DEVICES[device_name]


@pytest.fixture()
def dtype(dtype_name) -> torch.dtype:
    return TEST_DTYPES[dtype_name]


def pytest_generate_tests(metafunc):
    device_names = None
    dtype_names = None
    if 'device_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--device')
        if raw_value == 'all':
            device_names = list(TEST_DEVICES.keys())
        else:
            device_names = raw_value.split(',')
    if 'dtype_name' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype')
        if raw_value == 'all':
            dtype_names = list(TEST_DTYPES.keys())
        else:
            dtype_names = raw_value.split(',')
    if device_names is not None and dtype_names is not None:
        # Exclude any blacklisted device/dtype combinations.
        params = [combo for combo in product(device_names, dtype_names)
                  if combo not in DEVICE_DTYPE_BLACKLIST]
        metafunc.parametrize('device_name,dtype_name', params)
    elif device_names is not None:
        metafunc.parametrize('device_name', device_names)
    elif dtype_names is not None:
        metafunc.parametrize('dtype_name', dtype_names)


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--dtype', action="store", default="float32")
