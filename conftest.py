# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
from functools import partial
from itertools import product

import numpy as np
import pytest
import torch

import kornia
from kornia.utils._compat import torch_version

try:
    import torch._dynamo

    _backends_non_experimental = torch._dynamo.list_backends()
except ImportError:
    _backends_non_experimental = []


WEIGHTS_CACHE_DIR = "weights/"


def get_test_devices() -> dict[str, torch.device]:
    """Create a dictionary with the devices to test the source code.

    CUDA devices will be test only in case the current hardware supports it.

    Return:
        dict(str, torch.device): list with devices names.

    """
    devices: dict[str, torch.device] = {}
    devices["cpu"] = torch.device("cpu")
    if torch.cuda.is_available():
        devices["cuda"] = torch.device("cuda:0")
    if kornia.xla_is_available():
        import torch_xla.core.xla_model as xm

        devices["tpu"] = xm.xla_device()
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            devices["mps"] = torch.device("mps")
    return devices


def get_test_dtypes() -> dict[str, torch.dtype]:
    """Create a dictionary with the dtypes the source code.

    Return:
        dict(str, torch.dtype): list with dtype names.

    """
    dtypes: dict[str, torch.dtype] = {}
    dtypes["bfloat16"] = torch.bfloat16
    dtypes["float16"] = torch.float16
    dtypes["float32"] = torch.float32
    dtypes["float64"] = torch.float64
    return dtypes


# setup the devices to test the source code

TEST_DEVICES: dict[str, torch.device] = get_test_devices()
TEST_DTYPES: dict[str, torch.dtype] = get_test_dtypes()
TEST_OPTIMIZER_BACKEND = {"", None, "jit", *_backends_non_experimental}
# Combinations of device and dtype to be excluded from testing.
# DEVICE_DTYPE_BLACKLIST = {('cpu', 'float16')}
DEVICE_DTYPE_BLACKLIST = {}


@pytest.fixture()
def device(device_name) -> torch.device:
    """Return device for testing."""
    return TEST_DEVICES[device_name]


@pytest.fixture()
def dtype(dtype_name) -> torch.dtype:
    """Return dtype for testing."""
    return TEST_DTYPES[dtype_name]


@pytest.fixture()
def torch_optimizer(optimizer_backend):
    """Return torch optimizer."""
    if not optimizer_backend:
        return lambda x: x

    if optimizer_backend == "jit":
        return torch.jit.script

    if hasattr(torch, "compile") and sys.platform == "linux":
        if (not (sys.version_info[:2] == (3, 11) and torch_version() in {"2.0.0", "2.0.1"})) and (
            not sys.version_info[:2] == (3, 12)
        ):
            # torch compile don't have support for python3.12 yet
            torch._dynamo.reset()
            # torch compile just have support for python 3.11 after torch 2.1.0
            return partial(
                torch.compile, backend=optimizer_backend
            )  # TODO: explore the others parameters of torch compile

    pytest.skip(f"skipped because {torch.__version__} not have `compile` available! Failed to setup dynamo.")


def pytest_generate_tests(metafunc):
    """Generate tests."""
    device_names = None
    dtype_names = None
    optimizer_backends_names = None

    if "device_name" in metafunc.fixturenames:
        raw_value = metafunc.config.getoption("--device")
        if raw_value == "all":
            device_names = list(TEST_DEVICES.keys())
        else:
            device_names = raw_value.split(",")
    if "dtype_name" in metafunc.fixturenames:
        raw_value = metafunc.config.getoption("--dtype")
        if raw_value == "all":
            dtype_names = list(TEST_DTYPES.keys())
        else:
            dtype_names = raw_value.split(",")

    if "optimizer_backend" in metafunc.fixturenames:
        raw_value = metafunc.config.getoption("--optimizer")
        if raw_value == "all":
            optimizer_backends_names = TEST_OPTIMIZER_BACKEND
        else:
            optimizer_backends_names = raw_value.split(",")

    if device_names is not None and dtype_names is not None and optimizer_backends_names is not None:
        # Exclude any blacklisted device/dtype combinations.
        params = [
            combo
            for combo in product(device_names, dtype_names, optimizer_backends_names)
            if combo not in DEVICE_DTYPE_BLACKLIST
        ]
        metafunc.parametrize("device_name,dtype_name,optimizer_backend", params)
    elif device_names is not None and dtype_names is not None and optimizer_backends_names is None:
        # Exclude any blacklisted device/dtype combinations.
        params = [combo for combo in product(device_names, dtype_names) if combo not in DEVICE_DTYPE_BLACKLIST]
        metafunc.parametrize("device_name,dtype_name", params)
    elif device_names is not None and dtype_names is None and optimizer_backends_names is not None:
        params = product(device_names, optimizer_backends_names)
        metafunc.parametrize("device_name,optimizer_backend", params)

    elif device_names is not None:
        metafunc.parametrize("device_name", device_names)
    elif dtype_names is not None:
        metafunc.parametrize("dtype_name", dtype_names)
    elif optimizer_backends_names is not None:
        metafunc.parametrize("optimizer_backend", optimizer_backends_names)


def pytest_collection_modifyitems(config, items):
    """Collect test options."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add options."""
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--dtype", action="store", default="float32")
    parser.addoption("--optimizer", action="store", default="inductor")
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def _setup_torch_compile():
    if hasattr(torch, "compile") and sys.platform == "linux":
        print("Setting up torch compile...")
        torch.set_float32_matmul_precision("high")

        def _dummy_function(x, y):
            return (x + y).sum()

        class _dummy_module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return (x**2).sum()

        torch.compile(_dummy_function)
        torch.compile(_dummy_module())


def pytest_sessionstart(session):
    """Start pytest session."""
    try:
        _setup_torch_compile()
    except RuntimeError as ex:
        if "not yet supported for torch.compile" not in str(
            ex
        ) and "Dynamo is not supported on Python 3.12+" not in str(ex):
            raise ex

    os.makedirs(WEIGHTS_CACHE_DIR, exist_ok=True)
    torch.hub.set_dir(WEIGHTS_CACHE_DIR)
    # For HuggingFace model caching
    os.environ["HF_HOME"] = WEIGHTS_CACHE_DIR


def _get_env_info() -> dict[str, dict[str, str]]:
    if not hasattr(torch.utils, "collect_env"):
        return {}

    run_lmb = torch.utils.collect_env.run
    separator = ":"
    br = "\n"

    def _get_key_value(v: str):
        parts = v.split(separator)
        return parts[0].strip(), parts[-1].strip()

    def _get_cpu_info() -> dict[str, str]:
        cpu_info = {}
        cpu_str = torch.utils.collect_env.get_cpu_info(run_lmb)
        if not cpu_str:
            return {}

        for data in cpu_str.split(br):
            key, value = _get_key_value(data)
            cpu_info[key] = value

        return cpu_info

    def _get_gpu_info() -> dict[str, str]:
        gpu_info = {}
        gpu_str = torch.utils.collect_env.get_gpu_info(run_lmb)

        if not gpu_str:
            return {}

        for data in gpu_str.split(br):
            key, value = _get_key_value(data)
            gpu_info[key] = value

        return gpu_info

    return {
        "cpu": _get_cpu_info(),
        "gpu": _get_gpu_info(),
        "nvidia": torch.utils.collect_env.get_nvidia_driver_version(run_lmb),
        "gcc": torch.utils.collect_env.get_gcc_version(run_lmb),
    }


def pytest_report_header(config):
    """Return report header."""
    try:
        import accelerate

        accelerate_info = f"accelerate-{accelerate.__version__}"
    except ImportError:
        accelerate_info = "`accelerate` not found"

    import kornia_rs
    import onnx

    env_info = _get_env_info()
    CACHED_WEIGTHS = os.listdir(WEIGHTS_CACHE_DIR)
    if "cpu" in env_info:
        desired_cpu_info = ["Model name", "Architecture", "CPU(s)", "Thread(s) per core", "CPU max MHz", "CPU min MHz"]
        cpu_info = "cpu info:\n" + "\n".join(
            f"\t- {i}: {env_info['cpu'][i]}" for i in desired_cpu_info if i in env_info["cpu"]
        )
    else:
        cpu_info = ""
    gpu_info = f"gpu info: {env_info['gpu']}" if "gpu" in env_info else ""
    gcc_info = f"gcc info: {env_info['gcc']}" if "gcc" in env_info else ""

    return f"""
{cpu_info}
{gpu_info}
main deps:
    - kornia-{kornia.__version__}
    - torch-{torch.__version__}
        - commit: {torch.version.git_version}
        - cuda: {torch.version.cuda}
        - nvidia-driver: {env_info["nvidia"] if "nvidia" in env_info else None}
x deps:
    - {accelerate_info}
dev deps:
    - kornia_rs-{kornia_rs.__version__}
    - onnx-{onnx.__version__}
{gcc_info}
available optimizers: {TEST_OPTIMIZER_BACKEND}
model weights cached: {CACHED_WEIGTHS}
"""


@pytest.fixture(autouse=True)
def add_doctest_deps(doctest_namespace):
    """Add dependencies for doctests."""
    doctest_namespace["np"] = np
    doctest_namespace["torch"] = torch
    doctest_namespace["kornia"] = kornia


# the commit hash for the data version
sha: str = "cb8f42bf28b9f347df6afba5558738f62a11f28a"
sha2: str = "f7d8da661701424babb64850e03c5e8faec7ea62"
sha3: str = "8b98f44abbe92b7a84631ed06613b08fee7dae14"
sha4: str = "85bf178d7baeea2863e941b4badd9f1899ef3657"


@pytest.fixture(scope="session")
def data(request):
    """Return loaded data."""
    url = {
        "loftr_homo": f"https://github.com/kornia/data_test/blob/{sha}/loftr_outdoor_and_homography_data.pt?raw=true",
        "loftr_fund": f"https://github.com/kornia/data_test/blob/{sha}/loftr_indoor_and_fundamental_data.pt?raw=true",
        "adalam_idxs": f"https://github.com/kornia/data_test/blob/{sha2}/adalam_test.pt?raw=true",
        "lightglue_idxs": f"https://github.com/kornia/data_test/blob/{sha2}/adalam_test.pt?raw=true",
        "disk_outdoor": f"https://github.com/kornia/data_test/blob/{sha3}/knchurch_disk.pt?raw=true",
        "dexined": "https://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth",
        "eloftr_outdoor": f"https://github.com/AbhiKhoyani/kornia_data_test/blob/{sha4}/eloftr_outdoor_full_fp32_data.pt?raw=true"
    }
    return torch.hub.load_state_dict_from_url(url[request.param], map_location=torch.device("cpu"))
