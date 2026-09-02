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

from __future__ import annotations

import types

import pytest
import torch

from conftest import _is_subprocess_isolated_test, skip_half_precision_on_cuda

pytest_plugins = ["pytester"]

_skip_fixture_fn = getattr(skip_half_precision_on_cuda, "__wrapped__", skip_half_precision_on_cuda)


def _make_mock_item(params: dict | None, isolate: bool = True):
    config = types.SimpleNamespace(
        getoption=lambda opt, default=False: isolate if opt == "--isolate-half-precision" else default
    )
    callspec = types.SimpleNamespace(params=params) if params is not None else None
    return types.SimpleNamespace(config=config, callspec=callspec)


def _make_mock_request(
    fixturenames: list[str],
    fixture_values: dict[str, object],
    params: dict | None = None,
    isolate: bool = False,
):
    node = _make_mock_item(params, isolate=isolate)
    config = node.config

    def getfixturevalue(name: str):
        if name in fixture_values:
            return fixture_values[name]
        raise pytest.FixtureLookupError(name, node)

    return types.SimpleNamespace(
        fixturenames=fixturenames,
        getfixturevalue=getfixturevalue,
        config=config,
        node=node,
    )


class TestIsSubprocessIsolatedTest:
    def test_reproduce_issue_4126_local_half_dtype(self):
        """Issue #4126: local half_dtype on CUDA must be isolated even when parent job is float32."""
        item = _make_mock_item(
            {"device_name": "cuda", "dtype_name": "float32", "half_dtype": torch.float16},
            isolate=True,
        )
        assert _is_subprocess_isolated_test(item) is True

    @pytest.mark.parametrize("target_dtype", [torch.float16, torch.bfloat16])
    def test_local_custom_param_names(self, target_dtype):
        """Tests using custom parameter names (desc_dtype, image_dtype, etc.) must be isolated."""
        item = _make_mock_item(
            {"device_name": "cuda", "desc_dtype": target_dtype},
            isolate=True,
        )
        assert _is_subprocess_isolated_test(item) is True

    @pytest.mark.parametrize("target_dtype_name", ["float16", "bfloat16"])
    def test_global_dtype_fixture_preserved(self, target_dtype_name):
        """Existing global fixture tests using dtype_name must continue to be isolated."""
        item = _make_mock_item(
            {"device_name": "cuda", "dtype_name": target_dtype_name},
            isolate=True,
        )
        assert _is_subprocess_isolated_test(item) is True

    def test_negative_controls(self, monkeypatch):
        """Float32, CPU devices, unflagged runs, and subprocesses must NOT be isolated."""
        # Standard float32 test
        item_f32 = _make_mock_item({"device_name": "cuda", "dtype_name": "float32", "desc_dtype": torch.float32})
        assert _is_subprocess_isolated_test(item_f32) is False

        # CPU device with half precision
        item_cpu = _make_mock_item({"device_name": "cpu", "half_dtype": torch.float16})
        assert _is_subprocess_isolated_test(item_cpu) is False

        # Flag not set (--isolate-half-precision=False)
        item_no_flag = _make_mock_item({"device_name": "cuda", "half_dtype": torch.float16}, isolate=False)
        assert _is_subprocess_isolated_test(item_no_flag) is False

        # Inside subprocess (KORNIA_TEST_IN_SUBPROCESS=1)
        item_in_sub = _make_mock_item({"device_name": "cuda", "half_dtype": torch.float16}, isolate=True)
        monkeypatch.setenv("KORNIA_TEST_IN_SUBPROCESS", "1")
        assert _is_subprocess_isolated_test(item_in_sub) is False

        # No callspec
        item_no_callspec = _make_mock_item(None, isolate=True)
        assert _is_subprocess_isolated_test(item_no_callspec) is False


class TestSkipHalfPrecisionOnCuda:
    def test_local_half_dtype_is_skipped_without_isolation_flag(self):
        """Issue #4126: local half_dtype on CUDA without --isolate-half-precision must be skipped."""
        req = _make_mock_request(
            fixturenames=["device"],
            fixture_values={"device": torch.device("cuda:0")},
            params={"device_name": "cuda", "half_dtype": torch.float16},
            isolate=False,
        )
        with pytest.raises(pytest.skip.Exception, match="float16 on CUDA: skipped by default"):
            _skip_fixture_fn(req)

    def test_global_dtype_fixture_is_skipped_without_isolation_flag(self):
        """Global dtype fixture test on CUDA without isolation must be skipped."""
        req = _make_mock_request(
            fixturenames=["device", "dtype"],
            fixture_values={"device": torch.device("cuda:0"), "dtype": torch.bfloat16},
            params={"device_name": "cuda", "dtype_name": "bfloat16"},
            isolate=False,
        )
        with pytest.raises(pytest.skip.Exception, match="bfloat16 on CUDA: skipped by default"):
            _skip_fixture_fn(req)

    def test_not_skipped_on_cpu_or_float32(self):
        """Float32 and CPU tests must not be skipped."""
        # CPU test with half_dtype
        req_cpu = _make_mock_request(
            fixturenames=["device"],
            fixture_values={"device": torch.device("cpu")},
            params={"device_name": "cpu", "half_dtype": torch.float16},
            isolate=False,
        )
        assert _skip_fixture_fn(req_cpu) is None

        # CUDA test with float32
        req_f32 = _make_mock_request(
            fixturenames=["device", "dtype"],
            fixture_values={"device": torch.device("cuda:0"), "dtype": torch.float32},
            params={"device_name": "cuda", "dtype_name": "float32"},
            isolate=False,
        )
        assert _skip_fixture_fn(req_f32) is None


class TestIntegrationLocalHalfPrecision:
    def test_local_half_dtype_lifecycle_with_pytester(self, pytester):
        """Integration test: verify real pytest execution with local half_dtype."""
        test_file = pytester.makepyfile(
            """
            import pytest
            import torch

            @pytest.mark.parametrize("half_dtype", [torch.float16])
            def test_dummy(device, half_dtype):
                assert True
            """
        )

        # 1. CUDA + --dtype=float32 without --isolate-half-precision -> Must be SKIPPED
        result_cuda = pytester.runpytest_subprocess(
            "-p",
            "conftest",
            str(test_file),
            "-o",
            "testpaths=.",
            "--device=cuda",
            "--dtype=float32",
        )
        result_cuda.assert_outcomes(skipped=1)

        # 2. CPU + --dtype=float32 -> Must PASS
        result_cpu = pytester.runpytest_subprocess(
            "-p",
            "conftest",
            str(test_file),
            "-o",
            "testpaths=.",
            "--device=cpu",
            "--dtype=float32",
        )
        result_cpu.assert_outcomes(passed=1)


class TestDeviceAgnosticSelection:
    def test_device_agnostic_tests_run_once_in_cpu_containing_matrix(self, pytester):
        test_file = pytester.makepyfile(
            """
            import pytest

            @pytest.mark.device_agnostic
            def test_device_agnostic():
                assert True

            def test_device_specific(device):
                assert True
            """
        )

        result_cpu = pytester.runpytest_subprocess(
            "-p",
            "conftest",
            str(test_file),
            "-o",
            "testpaths=.",
            "--device=cpu",
            "--dtype=float32",
        )
        result_cpu.assert_outcomes(passed=2)

        result_cuda = pytester.runpytest_subprocess(
            "-p",
            "conftest",
            str(test_file),
            "-o",
            "testpaths=.",
            "--device=cuda",
            "--dtype=float32",
        )
        result_cuda.assert_outcomes(passed=1, deselected=1)
