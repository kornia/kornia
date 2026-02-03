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

import argparse
import os
from functools import partial
from itertools import product

import numpy as np
import pytest
import torch

import kornia

try:
    import torch._dynamo

    _backends_non_experimental = torch._dynamo.list_backends()
except ImportError:
    _backends_non_experimental = []


WEIGHTS_CACHE_DIR = "weights/"


def get_test_devices() -> dict[str, torch.device]:
    """Create a dictionary with the devices to test the source code.

    CUDA devices will be tested only if the current hardware supports it.

    Returns:
        Dictionary mapping device names to torch.device objects.
    """
    devices: dict[str, torch.device] = {"cpu": torch.device("cpu")}

    if torch.cuda.is_available():
        devices["cuda"] = torch.device("cuda:0")

    if kornia.core.utils.xla_is_available():
        import torch_xla.core.xla_model as xm

        devices["tpu"] = xm.xla_device()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices["mps"] = torch.device("mps")

    return devices


def get_test_dtypes() -> dict[str, torch.dtype]:
    """Create a dictionary with the dtypes to test.

    Returns:
        Dictionary mapping dtype names to torch.dtype objects.
    """
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }


# setup the devices to test the source code

TEST_DEVICES: dict[str, torch.device] = get_test_devices()
TEST_DTYPES: dict[str, torch.dtype] = get_test_dtypes()
TEST_OPTIMIZER_BACKEND = {"", None, "jit", *_backends_non_experimental}
# Combinations of device and dtype to be excluded from testing.
# Example: DEVICE_DTYPE_BLACKLIST = {('cpu', 'float16')}
DEVICE_DTYPE_BLACKLIST: set[tuple[str, ...]] = set()


@pytest.fixture()
def device(device_name) -> torch.device:
    """Return device for testing, skipping if device is unavailable."""
    if device_name not in TEST_DEVICES:
        pytest.skip(f"Device '{device_name}' is not available on this system")
    return TEST_DEVICES[device_name]


@pytest.fixture()
def dtype(dtype_name) -> torch.dtype:
    """Return dtype for testing."""
    return TEST_DTYPES[dtype_name]


@pytest.fixture()
def torch_optimizer(optimizer_backend):
    """Return torch optimizer based on backend selection.

    Args:
        optimizer_backend: The optimization backend ('jit', 'inductor', etc.)

    Returns:
        A function that optimizes/compiles torch modules or functions.
    """
    if not optimizer_backend:
        return lambda x: x

    if optimizer_backend == "jit":
        return torch.jit.script

    torch._dynamo.reset()
    return partial(torch.compile, backend=optimizer_backend)


def _parse_test_option(config, option: str, all_values: dict | set) -> list[str]:
    """Parse a test option from CLI, expanding 'all' to full list."""
    raw_value = config.getoption(option)
    if raw_value == "all":
        return list(all_values.keys()) if isinstance(all_values, dict) else list(all_values)
    return raw_value.split(",")


def pytest_generate_tests(metafunc) -> None:
    """Generate test parametrization based on fixtures and CLI options."""
    # Build list of (fixture_name, values) for fixtures that are used
    params: list[tuple[str, list]] = []

    if "device_name" in metafunc.fixturenames:
        params.append(("device_name", _parse_test_option(metafunc.config, "--device", TEST_DEVICES)))
    if "dtype_name" in metafunc.fixturenames:
        params.append(("dtype_name", _parse_test_option(metafunc.config, "--dtype", TEST_DTYPES)))
    if "optimizer_backend" in metafunc.fixturenames:
        params.append(("optimizer_backend", _parse_test_option(metafunc.config, "--optimizer", TEST_OPTIMIZER_BACKEND)))

    if not params:
        return

    # Single parameter: pass values directly (not as tuples)
    if len(params) == 1:
        name, values = params[0]
        metafunc.parametrize(name, values)
        return

    # Multiple parameters: generate combinations and filter blacklisted ones
    names = ",".join(name for name, _ in params)
    values = [v for _, v in params]
    combinations = [combo for combo in product(*values) if combo[:2] not in DEVICE_DTYPE_BLACKLIST]
    metafunc.parametrize(names, combinations)


def pytest_collection_modifyitems(config, items):
    """Collect test options."""
    # Deselect dynamo/compile tests when no optimizer is specified
    # Check environment variable directly (not config option which has default "inductor")
    optimizer_env = os.environ.get("KORNIA_TEST_OPTIMIZER", "").strip()
    if not optimizer_env:
        # Filter out tests with "dynamo" or "compile" in their name
        items[:] = [item for item in items if "dynamo" not in item.name.lower() and "compile" not in item.name.lower()]

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add options with environment variable fallbacks.

    Environment variables (for CI/pixi integration):
        KORNIA_TEST_DEVICE: Device to test on (default: cpu)
        KORNIA_TEST_DTYPE: Data type to test (default: float32)
        KORNIA_TEST_OPTIMIZER: Optimizer backend (default: inductor)
        KORNIA_TEST_RUNSLOW: Run slow tests (default: false)
    """
    options = [
        (
            "--device",
            {
                "action": "store",
                "default": os.environ.get("KORNIA_TEST_DEVICE", "cpu"),
                "help": "Device to run tests on (env: KORNIA_TEST_DEVICE)",
            },
        ),
        (
            "--dtype",
            {
                "action": "store",
                "default": os.environ.get("KORNIA_TEST_DTYPE", "float32"),
                "help": "Data type for tests (env: KORNIA_TEST_DTYPE)",
            },
        ),
        (
            "--optimizer",
            {
                "action": "store",
                "default": os.environ.get("KORNIA_TEST_OPTIMIZER", "inductor"),
                "help": "Optimizer backend (env: KORNIA_TEST_OPTIMIZER)",
            },
        ),
        (
            "--runslow",
            {
                "action": "store_true",
                "default": os.environ.get("KORNIA_TEST_RUNSLOW", "false").lower() == "true",
                "help": "Run slow tests (env: KORNIA_TEST_RUNSLOW)",
            },
        ),
    ]

    for name, kwargs in options:
        try:
            parser.addoption(name, **kwargs)
        except (argparse.ArgumentError, ValueError):
            pass


def _setup_torch_compile() -> None:
    """Warm up torch.compile to reduce first-run latency in tests."""
    print("Setting up torch compile...")
    torch.set_float32_matmul_precision("high")

    def _dummy_fn(x, y):
        return (x + y).sum()

    class _DummyModule(torch.nn.Module):
        def forward(self, x):
            return (x**2).sum()

    torch.compile(_dummy_fn)
    torch.compile(_DummyModule())


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
    cached_weights = os.listdir(WEIGHTS_CACHE_DIR) if os.path.exists(WEIGHTS_CACHE_DIR) else []
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
model weights cached: {cached_weights}
"""


@pytest.fixture(autouse=True)
def add_doctest_deps(doctest_namespace):
    """Add dependencies for doctests."""
    doctest_namespace["np"] = np
    doctest_namespace["torch"] = torch
    doctest_namespace["kornia"] = kornia


# Test data commit hashes from kornia/data_test repository
_DATA_TEST_SHA = {
    "loftr": "cb8f42bf28b9f347df6afba5558738f62a11f28a",
    "adalam": "f7d8da661701424babb64850e03c5e8faec7ea62",
    "disk": "8b98f44abbe92b7a84631ed06613b08fee7dae14",
}

# URLs for test data files
_TEST_DATA_URLS: dict[str, str] = {
    "loftr_homo": f"https://github.com/kornia/data_test/blob/{_DATA_TEST_SHA['loftr']}/loftr_outdoor_and_homography_data.pt?raw=true",
    "loftr_fund": f"https://github.com/kornia/data_test/blob/{_DATA_TEST_SHA['loftr']}/loftr_indoor_and_fundamental_data.pt?raw=true",
    "adalam_idxs": f"https://github.com/kornia/data_test/blob/{_DATA_TEST_SHA['adalam']}/adalam_test.pt?raw=true",
    "lightglue_idxs": f"https://github.com/kornia/data_test/blob/{_DATA_TEST_SHA['adalam']}/adalam_test.pt?raw=true",
    "disk_outdoor": f"https://github.com/kornia/data_test/blob/{_DATA_TEST_SHA['disk']}/knchurch_disk.pt?raw=true",
    "dexined": "https://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth",
}


@pytest.fixture(scope="session")
def data(request):
    """Load test data from remote URL.

    Use with @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    """
    if request.param not in _TEST_DATA_URLS:
        raise ValueError(f"Unknown test data: {request.param}. Available: {list(_TEST_DATA_URLS.keys())}")
    return torch.hub.load_state_dict_from_url(_TEST_DATA_URLS[request.param], map_location=torch.device("cpu"))
