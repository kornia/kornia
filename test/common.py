import torch
import pytest


def get_test_devices():
    """Creates a string list with the devices type to test the source code.
    CUDA devices will be test only in case the current hardware supports it.

    Return:
        list(str): list with devices names.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


# setup the devices to test the source code

TEST_DEVICES = get_test_devices()


@pytest.fixture()
def device_type(request):
    print("Executing device type")
    typ = request.config.getoption('--typetest')
    return typ
