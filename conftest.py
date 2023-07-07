import sys
from itertools import product
from typing import Dict

import numpy
import pytest
import torch

import kornia


def get_test_devices() -> Dict[str, torch.device]:
    """Create a dictionary with the devices to test the source code. CUDA devices will be test only in case the
    current hardware supports it.

    Return:
        dict(str, torch.device): list with devices names.
    """
    devices: Dict[str, torch.device] = {}
    devices["cpu"] = torch.device("cpu")
    if torch.cuda.is_available():
        devices["cuda"] = torch.device("cuda:0")
    if kornia.xla_is_available():
        import torch_xla.core.xla_model as xm

        devices["tpu"] = xm.xla_device()
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            devices["mps"] = torch.device("mps")
    return devices


def get_test_dtypes() -> Dict[str, torch.dtype]:
    """Create a dictionary with the dtypes the source code.

    Return:
        dict(str, torch.dtype): list with dtype names.
    """
    dtypes: Dict[str, torch.dtype] = {}
    dtypes["bfloat16"] = torch.bfloat16
    dtypes["float16"] = torch.float16
    dtypes["float32"] = torch.float32
    dtypes["float64"] = torch.float64
    return dtypes


# setup the devices to test the source code

TEST_DEVICES: Dict[str, torch.device] = get_test_devices()
TEST_DTYPES: Dict[str, torch.dtype] = get_test_dtypes()

# Combinations of device and dtype to be excluded from testing.
# DEVICE_DTYPE_BLACKLIST = {('cpu', 'float16')}
DEVICE_DTYPE_BLACKLIST = {}


@pytest.fixture()
def device(device_name) -> torch.device:
    return TEST_DEVICES[device_name]


@pytest.fixture()
def dtype(dtype_name) -> torch.dtype:
    return TEST_DTYPES[dtype_name]


@pytest.fixture(scope='session')
def torch_optimizer():
    if hasattr(torch, 'compile') and sys.platform == "linux":
        torch.set_float32_matmul_precision('high')
        return torch.compile

    pytest.skip(f"skipped because {torch.__version__} not have `compile` available! Failed to setup dynamo.")


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
        params = [combo for combo in product(device_names, dtype_names) if combo not in DEVICE_DTYPE_BLACKLIST]
        metafunc.parametrize('device_name,dtype_name', params)
    elif device_names is not None:
        metafunc.parametrize('device_name', device_names)
    elif dtype_names is not None:
        metafunc.parametrize('dtype_name', dtype_names)


def pytest_addoption(parser):
    parser.addoption('--device', action="store", default="cpu")
    parser.addoption('--dtype', action="store", default="float32")


@pytest.fixture(autouse=True)
def add_doctest_deps(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["torch"] = torch
    doctest_namespace["kornia"] = kornia


# the commit hash for the data version
sha: str = 'cb8f42bf28b9f347df6afba5558738f62a11f28a'
sha2: str = 'f7d8da661701424babb64850e03c5e8faec7ea62'
sha3: str = '8b98f44abbe92b7a84631ed06613b08fee7dae14'


@pytest.fixture(scope='session')
def data(request):
    url = {
        'loftr_homo': f'https://github.com/kornia/data_test/blob/{sha}/loftr_outdoor_and_homography_data.pt?raw=true',
        'loftr_fund': f'https://github.com/kornia/data_test/blob/{sha}/loftr_indoor_and_fundamental_data.pt?raw=true',
        'adalam_idxs': f'https://github.com/kornia/data_test/blob/{sha2}/adalam_test.pt?raw=true',
        'lightglue_idxs': f'https://github.com/kornia/data_test/blob/{sha2}/adalam_test.pt?raw=true',
        'disk_outdoor': f'https://github.com/kornia/data_test/blob/{sha3}/knchurch_disk.pt?raw=true',
        'dexined': 'https://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth',
    }
    return torch.hub.load_state_dict_from_url(url[request.param], map_location=torch.device('cpu'))
