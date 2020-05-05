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
def device(request) -> torch.device:
    _device_type: str = request.config.getoption('--typetest')
    return TEST_DEVICES[_device_type]


@pytest.fixture()
def dtype(request) -> torch.dtype:
    _dtype_name: str = request.config.getoption('--dtypetest')
    return TEST_DTYPES[_dtype_name]
