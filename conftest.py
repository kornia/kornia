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
import subprocess
import sys
import time
from functools import partial
from itertools import product

import numpy as np
import pytest
import torch

try:
    from pytest import TestReport  # public since pytest 7.x
except ImportError:  # pragma: no cover
    from _pytest.reports import TestReport  # type: ignore[no-redef]

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
        devices["mps"] = torch.device("mps:0")

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

    # MPS does not support float64; gradcheck requires float64 — skip all gradcheck tests on MPS
    skip_mps_gradcheck = pytest.mark.skip(reason="gradcheck requires float64 which is not supported on MPS")
    for item in items:
        if "gradcheck" in item.name.lower() and "[mps" in item.nodeid:
            item.add_marker(skip_mps_gradcheck)

    # MPS does not support complex128 (cdouble); skip tests parametrized with it
    skip_mps_cdouble = pytest.mark.skip(reason="MPS does not support complex128 (cdouble)")
    for item in items:
        if "[mps" in item.nodeid and "cdtype1" in item.nodeid:
            item.add_marker(skip_mps_cdouble)

    # MPS autocast uses float16 and does not preserve original dtype — skip autocast tests on MPS
    skip_mps_autocast = pytest.mark.skip(reason="MPS autocast changes dtype to float16, not supported the same way")
    for item in items:
        if "autocast" in item.name.lower() and "[mps" in item.nodeid:
            item.add_marker(skip_mps_autocast)

    tf32_enabled = config.getoption("--tf32")

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Tests marked @pytest.mark.tf32 are known to produce incorrect results when
    # TF32 is active (torch.set_float32_matmul_precision("high")).  When running
    # without --tf32 (the default), mark them xfail so the suite stays green.
    # When --tf32 is explicitly passed, run them normally so the failures are visible.
    if not tf32_enabled:
        xfail_tf32 = pytest.mark.xfail(
            reason=(
                "This test is sensitive to TF32 (TensorFloat-32) precision reduction in CUDA matrix "
                "multiplications.  Run with --tf32 to reproduce the failure."
            ),
            strict=False,
        )
        for item in items:
            if "tf32" in item.keywords:
                item.add_marker(xfail_tf32)


def pytest_addoption(parser):
    """Add options with environment variable fallbacks.

    Environment variables (for CI/pixi integration):
        KORNIA_TEST_DEVICE: Device to test on (default: cpu)
        KORNIA_TEST_DTYPE: Data type to test (default: float32)
        KORNIA_TEST_OPTIMIZER: Optimizer backend (default: inductor)
        KORNIA_TEST_RUNSLOW: Run slow tests (default: false)
        KORNIA_TEST_TF32: Enable TF32 (TensorFloat-32) mode for CUDA matrix multiplications (default: false)
    """
    parser.addoption(
        "--device",
        action="store",
        default=os.environ.get("KORNIA_TEST_DEVICE", "cpu"),
        help="Device to run tests on (env: KORNIA_TEST_DEVICE)",
    )
    parser.addoption(
        "--dtype",
        action="store",
        default=os.environ.get("KORNIA_TEST_DTYPE", "float32"),
        help="Data type for tests (env: KORNIA_TEST_DTYPE)",
    )
    parser.addoption(
        "--optimizer",
        action="store",
        default=os.environ.get("KORNIA_TEST_OPTIMIZER", "inductor"),
        help="Optimizer backend (env: KORNIA_TEST_OPTIMIZER)",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=os.environ.get("KORNIA_TEST_RUNSLOW", "false").lower() == "true",
        help="Run slow tests (env: KORNIA_TEST_RUNSLOW)",
    )
    parser.addoption(
        "--tf32",
        action="store_true",
        default=os.environ.get("KORNIA_TEST_TF32", "false").lower() == "true",
        help=(
            "Enable TF32 (TensorFloat-32) mode for CUDA matrix multiplications "
            "(torch.set_float32_matmul_precision('high')). "
            "Tests marked @pytest.mark.tf32 are expected to fail under TF32 and are skipped unless this flag is set. "
            "(env: KORNIA_TEST_TF32)"
        ),
    )
    parser.addoption(
        "--isolate-half-precision",
        action="store_true",
        default=os.environ.get("KORNIA_TEST_ISOLATE_HALF", "false").lower() == "true",
        help=(
            "Run float16/bfloat16 CUDA tests in fresh subprocesses via subprocess.run. "
            "Each test gets its own Python process with no shared CUDA state, so a "
            "device-side assert cannot contaminate subsequent tests. "
            "Without this flag, float16/bfloat16 CUDA tests are skipped. "
            "(env: KORNIA_TEST_ISOLATE_HALF)"
        ),
    )


def _setup_torch_compile() -> None:
    """Warm up torch.compile to reduce first-run latency in tests."""
    print("Setting up torch compile...")

    def _dummy_fn(x, y):
        return (x + y).sum()

    class _DummyModule(torch.nn.Module):
        def forward(self, x):
            return (x**2).sum()

    torch.compile(_dummy_fn)
    torch.compile(_DummyModule())


def pytest_sessionstart(session):
    """Start pytest session."""
    # Enable TF32 only when explicitly requested via --tf32 / KORNIA_TEST_TF32=true.
    #
    # TF32 (TensorFloat-32) truncates float32 inputs to a 10-bit mantissa before
    # CUDA matrix multiplications (torch.bmm, torch.mm, etc.), giving ~float16
    # mantissa precision for those ops.  This can cause test failures for
    # numerically sensitive operations even though the same test passes without TF32.
    # By default we run with full float32 precision so that tests are deterministic
    # and correct.  Use --tf32 to reproduce the behaviour of torch.compile("high")
    # and to run the @pytest.mark.tf32-marked tests.
    if session.config.getoption("--tf32"):
        torch.set_float32_matmul_precision("high")

    # Skip torch.compile warmup in subprocess mode — it adds startup overhead and
    # pollutes the captured output used for failure reporting in pytest_runtest_protocol.
    if not os.environ.get("KORNIA_TEST_IN_SUBPROCESS"):
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


def _extract_failure_output(output: str) -> str:
    """Return just the FAILURES/ERRORS section from pytest stdout, or full output as fallback."""
    import re

    m = re.search(r"^=+ (FAILURES|ERRORS) =+", output, re.MULTILINE)
    if m:
        return output[m.start() :].strip()
    return output.strip()


def _is_subprocess_isolated_test(item) -> bool:
    """Return True if this test should be run in a fresh subprocess.

    Checks that:
    - ``--isolate-half-precision`` is set
    - we are NOT already inside a subprocess (``KORNIA_TEST_IN_SUBPROCESS`` env var)
    - the test is parametrised with a half-precision dtype on CUDA
    """
    if os.environ.get("KORNIA_TEST_IN_SUBPROCESS"):
        return False
    if not item.config.getoption("--isolate-half-precision", default=False):
        return False
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    params = callspec.params
    if params.get("dtype_name") not in ("float16", "bfloat16"):
        return False
    if params.get("device_name") != "cuda":
        return False
    return True


def pytest_runtest_protocol(item, nextitem):
    """Run float16/bfloat16 CUDA tests in a fresh subprocess for true isolation.

    ``pytest-forked`` uses ``fork()``, which copies the parent's CUDA context handle
    into the child.  A device-side assert in the child corrupts the *same* underlying
    GPU state the parent holds — so the isolation is illusory for CUDA float16.

    This hook uses ``subprocess.run`` instead, which spawns a completely independent
    Python interpreter with no shared CUDA state.  The child's result (pass / fail /
    skip) is parsed and reported back into the parent's session as a synthetic
    ``TestReport``.
    """
    if not _is_subprocess_isolated_test(item):
        return None  # use the default protocol

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

    # Forward device/dtype so pytest_generate_tests in the subprocess produces the
    # same parametrisation as the parent — without these the [cuda-float16] nodeid
    # can't be found because the subprocess defaults to [cpu-float32].
    params = item.callspec.params
    device_name = params.get("device_name", "cpu")
    dtype_name = params.get("dtype_name", "float32")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        item.nodeid,
        "--no-header",
        "--tb=short",
        "-q",
        "--color=no",
        f"--device={device_name}",
        f"--dtype={dtype_name}",
    ]
    if item.config.getoption("--runslow"):
        cmd.append("--runslow")
    if item.config.getoption("--tf32"):
        cmd.append("--tf32")
    optimizer_backend = params.get("optimizer_backend")
    if optimizer_backend:
        cmd.append(f"--optimizer={optimizer_backend}")

    env = {**os.environ, "KORNIA_TEST_IN_SUBPROCESS": "1"}
    t0 = time.monotonic()
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, cwd=str(item.config.rootdir), env=env, check=False
    )
    duration = time.monotonic() - t0
    output = (proc.stdout + proc.stderr).strip()

    # exit code 5 → no tests collected (test was deselected or already parametrised away)
    if proc.returncode == 5:
        outcome: str = "skipped"
        longrepr = ("", 0, "subprocess: no tests collected")
    elif proc.returncode == 0:
        # Distinguish a genuine pass from a skipped test
        if "passed" not in output and "skipped" in output:
            skip_line = next(
                (ln.strip() for ln in output.splitlines() if "SKIP" in ln.upper()), "skipped in subprocess"
            )
            outcome = "skipped"
            longrepr = ("", 0, skip_line)
        else:
            outcome = "passed"
            longrepr = None
    else:
        outcome = "failed"
        longrepr = _extract_failure_output(output)

    def _report(when: str, out: str, rep_longrepr, dur: float = 0.0) -> TestReport:
        return TestReport(
            nodeid=item.nodeid,
            location=item.location,
            keywords=dict(item.keywords),
            outcome=out,
            longrepr=rep_longrepr,
            when=when,
            duration=dur,
        )

    for rep in [
        _report("setup", "passed", None),
        _report("call", outcome, longrepr, duration),
        _report("teardown", "passed", None),
    ]:
        item.ihook.pytest_runtest_logreport(report=rep)

    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


@pytest.fixture(autouse=True)
def skip_half_precision_on_cuda(request):
    """Skip float16/bfloat16 CUDA tests unless running inside a subprocess.

    CUDA device-side asserts are asynchronous: a failing half-precision kernel does not
    raise immediately but corrupts the CUDA context until the next synchronisation point,
    which may be inside a completely different (float32) test.  Once triggered, the
    context is permanently broken for the process — all subsequent CUDA ops fail.

    Default behaviour (no flag): float16/bfloat16 CUDA tests are *skipped*.

    With ``--isolate-half-precision`` (or ``KORNIA_TEST_ISOLATE_HALF=true``): each
    float16/bfloat16 CUDA test is intercepted by ``pytest_runtest_protocol`` *before*
    any fixture runs and executed in a fresh ``subprocess.run`` process.  This fixture
    only runs inside those subprocesses (where ``KORNIA_TEST_IN_SUBPROCESS=1`` is set)
    and exits immediately so the test proceeds normally.

    Usage::

        pytest tests/color/ --device=cuda --dtype=bfloat16 --isolate-half-precision
        pytest tests/       --device=cuda --dtype=all      --isolate-half-precision
    """
    # Inside a subprocess spawned by pytest_runtest_protocol — run the test normally.
    if os.environ.get("KORNIA_TEST_IN_SUBPROCESS"):
        return

    if "dtype" not in request.fixturenames:
        return
    dtype = request.getfixturevalue("dtype")
    if dtype not in (torch.bfloat16, torch.float16):
        return
    if "device" not in request.fixturenames:
        return

    try:
        device = request.getfixturevalue("device")
    except pytest.FixtureLookupError:
        return

    if device.type != "cuda":
        return

    if not request.config.getoption("--isolate-half-precision"):
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        pytest.skip(
            f"{dtype_name} on CUDA: skipped by default to prevent device-side assert contamination. "
            "Run with --isolate-half-precision to execute in isolated subprocesses."
        )


@pytest.fixture(autouse=True)
def cuda_device_assert_guard(request):
    """Guard against CUDA device-side assert contamination between tests.

    Active only when running inside a subprocess (``KORNIA_TEST_IN_SUBPROCESS=1``)
    or with ``--isolate-half-precision``, so regular float32 CI is not slowed
    down by the extra host-device synchronisations.

    This fixture synchronises CUDA before each test; if the context is already
    corrupted the test is skipped rather than allowed to fail spuriously.
    After each test a second synchronisation drains the queue so any async
    device-side assert surfaces in the test that caused it, not the next one.
    If a device-side assert is detected in the post-test sync the test is
    failed (not silently passed) so asynchronous errors are always visible.
    """
    in_subprocess = os.environ.get("KORNIA_TEST_IN_SUBPROCESS")
    isolate = request.config.getoption("--isolate-half-precision", default=False)
    if not (in_subprocess or isolate):
        yield
        return

    if "device" not in request.fixturenames:
        yield
        return

    try:
        device = request.getfixturevalue("device")
    except pytest.FixtureLookupError:
        yield
        return

    if device.type != "cuda":
        yield
        return

    # Pre-test: verify the CUDA context is healthy.
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        pytest.skip("CUDA context corrupted by a device-side assert in a previous test; run this test in isolation")

    yield

    # Post-test: drain the CUDA queue so any async device-side assert surfaces here,
    # in the test that caused it, rather than at the start of the next test.
    # Fail the test if a device-side assert is detected so it is not silently passed.
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        pytest.fail(f"CUDA device-side assert triggered during this test: {exc}")


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
