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


# setup the devices to test the source code

TEST_DEVICES: Dict[str, torch.device] = get_test_devices()


@pytest.fixture()
def device(request) -> torch.device:
    _device_type: str = request.config.getoption('--typetest')
    return TEST_DEVICES[_device_type]
