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

import random
import re
import shutil
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import conftest as project_conftest

from testing.half_precision_ci import (
    ManifestEntry,
    ManifestEnvironment,
    current_environment,
    get_profile,
    parse_manifest,
    seed_test_rng,
    serialize_manifest,
)

pytest_plugins = ["pytester"]


def _environment(**overrides: str) -> ManifestEnvironment:
    values = {
        "python": "3.11",
        "pytorch": "2.9.1",
        "os": "Linux",
        "arch": "x86_64",
        "pytest": "9.0.3",
        "numpy": "2.3.5",
        "pillow": "12.0.0",
    }
    values.update(overrides)
    return ManifestEnvironment(**values)


class TestManifestCodec:
    def test_round_trips_phase_and_opaque_exception_identity(self) -> None:
        profile = get_profile("cpu-float16")
        entries = [
            ManifestEntry("setup", "_pytest.outcomes.Failed", "tests/a.py::test_pytest_fail"),
            ManifestEntry("call", "some_test.LocalError", "tests/a.py::test_local[other-bfloat16-text]"),
        ]

        text = serialize_manifest(profile, entries, _environment(), "pytest tests/")
        parsed = parse_manifest(profile, text, _environment())

        assert parsed == {entry.nodeid: entry for entry in entries}
        assert "setup\t_pytest.outcomes.Failed\ttests/a.py::test_pytest_fail" in text
        assert "call\tsome_test.LocalError\ttests/a.py::test_local[other-bfloat16-text]" in text

    @pytest.mark.parametrize("phase", ["", "collect", "teardown"])
    def test_rejects_unsupported_phase(self, phase: str) -> None:
        profile = get_profile("cpu-float16")
        text = serialize_manifest(
            profile,
            [ManifestEntry("call", "builtins.AssertionError", "tests/a.py::test_a")],
            _environment(),
            "pytest tests/",
        ).replace("call\tbuiltins.AssertionError", f"{phase}\tbuiltins.AssertionError")

        with pytest.raises(ValueError, match="unsupported phase"):
            parse_manifest(profile, text, _environment())

    def test_rejects_duplicate_node_ids(self) -> None:
        profile = get_profile("cpu-float16")
        entry = ManifestEntry("call", "builtins.AssertionError", "tests/a.py::test_a")
        text = serialize_manifest(profile, [entry], _environment(), "pytest tests/")
        text += "call\tbuiltins.RuntimeError\ttests/a.py::test_a\n"

        with pytest.raises(ValueError, match="duplicate node ID"):
            parse_manifest(profile, text, _environment())

    def test_rejects_wrong_profile_and_enforced_environment(self) -> None:
        profile = get_profile("cpu-float16")
        text = serialize_manifest(profile, [], _environment(), "pytest tests/")

        with pytest.raises(ValueError, match=r"profile.*cpu-bfloat16"):
            parse_manifest(get_profile("cpu-bfloat16"), text, _environment())
        with pytest.raises(ValueError, match=r"Python.*3.12"):
            parse_manifest(profile, text, _environment(python="3.12"))

    def test_provenance_difference_warns_without_rejecting(self) -> None:
        profile = get_profile("cpu-float16")
        text = serialize_manifest(profile, [], _environment(pytest="8.4.2"), "pytest tests/")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert parse_manifest(profile, text, _environment(pytest="9.0.3")) == {}

        assert any("pytest provenance differs" in str(item.message) for item in caught)

    def test_focus_warns_but_complete_rejects_compatibility_difference(self, tmp_path: Path) -> None:
        from testing.half_precision_ci import KnownFailureTracker

        profile = get_profile("cpu-float16")
        environment = current_environment()
        path = tmp_path / "cpu_float16.txt"
        path.write_text(
            serialize_manifest(profile, [], replace(environment, python="0.0"), "pytest tests/"),
            encoding="utf-8",
        )

        with pytest.warns(UserWarning, match="Python compatibility differs"):
            KnownFailureTracker(profile, path, mode="focus")
        with pytest.raises(ValueError, match="Python mismatch"):
            KnownFailureTracker(profile, path, mode="complete")


def _write_profile_manifest(directory: Path, phase: str, exception: str, nodeid: str) -> Path:
    profile = get_profile("cpu-float16")
    path = directory / "cpu_float16.txt"
    path.write_text(
        serialize_manifest(
            profile,
            [ManifestEntry(phase, exception, nodeid)],  # type: ignore[arg-type]
            current_environment(),
            "pytest tests/",
        ),
        encoding="utf-8",
    )
    return path


def _rng_states() -> tuple[object, tuple, torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(),  # noqa: NPY002 - verify the process-global state is unchanged.
        torch.random.get_rng_state(),
    )


def _assert_rng_states_equal(
    first: tuple[object, tuple, torch.Tensor], second: tuple[object, tuple, torch.Tensor]
) -> None:
    assert first[0] == second[0]
    assert first[1][0] == second[1][0]
    assert np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])


def test_seed_test_rng_is_pure_stable_and_node_specific() -> None:
    before = _rng_states()

    first = seed_test_rng("tests/a.py::test_a[cpu-float16]")
    other = seed_test_rng("tests/a.py::test_b[cpu-float16]")
    repeated = seed_test_rng("tests/a.py::test_a[cpu-float16]")

    assert repeated == first
    assert other != first
    assert first == 495_473_945
    _assert_rng_states_equal(before, _rng_states())


def test_profile_rng_isolation_restores_all_global_states() -> None:
    before = _rng_states()

    with project_conftest._isolated_test_rng(seed_test_rng("tests/a.py::test_a[cpu-float16]")):
        random.random()
        np.random.random()  # noqa: NPY002
        torch.rand(())

    _assert_rng_states_equal(before, _rng_states())


def test_rng_seed_fixture_is_available_without_a_profile(test_rng_seed: int, request: pytest.FixtureRequest) -> None:
    assert test_rng_seed == seed_test_rng(request.node.nodeid)


def test_eager_rng_audit_matches_exact_inventory() -> None:
    """Validate audited (path, line, name) triples as a multiset, so a stale line still fails."""
    from testing.half_precision_eager_rng import AUDITED_EAGER_RNG_CALLS, find_eager_rng_calls

    discovered = Counter(
        (call.path, call.line, call.name) for call in find_eager_rng_calls(Path(project_conftest.__file__).parent)
    )
    audited = Counter((entry.call.path, entry.call.line, entry.call.name) for entry in AUDITED_EAGER_RNG_CALLS)

    stale = sorted((audited - discovered).elements())
    missing = sorted((discovered - audited).elements())
    assert not stale, f"audited entries with no matching call site (stale line?): {stale}"
    assert not missing, f"eager RNG call sites missing from the audit: {missing}"


def test_eager_rng_scanner_skips_lazy_function_and_lambda_bodies(tmp_path: Path) -> None:
    from testing.half_precision_eager_rng import EagerRngCall, find_eager_rng_calls

    (tmp_path / "test_sample.py").write_text(
        """
import torch

module_value = torch.rand(1)

class TestSample:
    class_value = torch.randn(1)

    @pytest.mark.parametrize("value", [torch.randint(0, 2, (1,))])
    def test_case(self, value=torch.randperm(2)):
        torch.rand(1)

lazy = lambda: torch.randn(1)
""",
        encoding="utf-8",
    )

    assert find_eager_rng_calls(tmp_path) == (
        EagerRngCall("test_sample.py", 4, "torch.rand"),
        EagerRngCall("test_sample.py", 7, "torch.randn"),
        EagerRngCall("test_sample.py", 9, "torch.randint"),
        EagerRngCall("test_sample.py", 10, "torch.randperm"),
    )


def test_eager_rng_node_coverage_is_boundary_aware() -> None:
    from testing.half_precision_eager_rng import eager_rng_calls_for_node

    prefix = "tests/enhance/test_core.py::TestAddWeighted::test_shape"
    assert len(eager_rng_calls_for_node(prefix)) == 3
    assert len(eager_rng_calls_for_node(f"{prefix}[cpu-float16-case]")) == 3
    assert eager_rng_calls_for_node(f"{prefix}_mismatch[cpu-float16]") == ()
    assert eager_rng_calls_for_node("tests/core/test_check.py::TestCheckShape::test_valid[cpu-float16]") == ()


def _install_candidate_recorder(pytester: pytest.Pytester) -> Path:
    output = pytester.path / "candidate.txt"
    pytester.makeconftest(
        """
        from pathlib import Path

        from testing.half_precision_ci import FailureRecorder, get_profile


        def pytest_configure(config):
            recorder = FailureRecorder(
                get_profile("cpu-float16"),
                Path(__file__).with_name("candidate.txt"),
                rootpath=config.rootpath,
                previous_entries={},
            )
            config.pluginmanager.register(recorder, "half-precision-failure-recorder")
        """
    )
    return output


def test_failure_recorder_writes_complete_v1_candidate_in_nodeid_order(pytester: pytest.Pytester) -> None:
    output = _install_candidate_recorder(pytester)
    pytester.makepyfile(
        test_sample="""
        import pytest


        def test_z_failure():
            raise ValueError("second after sorting")


        def test_a_failure():
            raise AssertionError("first after sorting")


        @pytest.fixture
        def failed_setup():
            raise RuntimeError("fixture setup failure")


        def test_setup_failure(failed_setup):
            pass


        def test_kornia_failure():
            from kornia.core.exceptions import ShapeError

            raise ShapeError("kornia validation failure")


        def test_skip():
            pytest.skip("not a failure")


        @pytest.mark.xfail(reason="already tracked")
        def test_existing_xfail():
            raise RuntimeError("not a new failure")
        """
    )

    result = pytester.runpytest("-q")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    recorded = output.read_text(encoding="utf-8")
    entries = parse_manifest(get_profile("cpu-float16"), recorded, current_environment(), source=output)
    assert list(entries.values()) == [
        ManifestEntry("call", "builtins.AssertionError", "test_sample.py::test_a_failure"),
        ManifestEntry("call", "kornia.core.exceptions.ShapeError", "test_sample.py::test_kornia_failure"),
        ManifestEntry("setup", "builtins.RuntimeError", "test_sample.py::test_setup_failure"),
        ManifestEntry("call", "builtins.ValueError", "test_sample.py::test_z_failure"),
    ]
    assert "NEW: call\tbuiltins.AssertionError\ttest_sample.py::test_a_failure" in result.stdout.str()


@pytest.mark.parametrize(
    "scenario", ["collection-error", "collect-only", "early-stop", "interrupt", "relevant-teardown"]
)
def test_failure_recorder_preserves_existing_candidate_on_incomplete_run(
    pytester: pytest.Pytester, scenario: str
) -> None:
    output = _install_candidate_recorder(pytester)
    output.write_text("preserve me\n", encoding="utf-8")
    if scenario == "collection-error":
        pytester.makepyfile(test_sample="raise RuntimeError('import failure')")
        args = ("-q",)
    elif scenario == "collect-only":
        pytester.makepyfile(test_sample="def test_ok(): pass")
        args = ("--collect-only", "-q")
    elif scenario == "early-stop":
        pytester.makepyfile(test_sample="def test_a(): assert False\ndef test_b(): pass")
        args = ("-x", "-q")
    elif scenario == "interrupt":
        pytester.makepyfile(test_sample="def test_interrupt(): raise KeyboardInterrupt()")
        args = ("-q",)
    else:
        pytester.makepyfile(
            test_sample="""
            import pytest

            @pytest.fixture
            def broken_teardown():
                yield
                raise RuntimeError("teardown")

            def test_failure(broken_teardown):
                raise AssertionError("call")
            """
        )
        args = ("-q",)

    if scenario == "interrupt":
        pytester.runpytest_subprocess(*args)
    else:
        pytester.runpytest(*args)

    assert output.read_text(encoding="utf-8") == "preserve me\n"


def test_failure_recorder_aborted_clean_run_fails_session(pytester: pytest.Pytester) -> None:
    _install_candidate_recorder(pytester)
    pytester.makepyfile(test_sample="def test_ok(): pass")

    result = pytester.runpytest("--collect-only", "-q")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "known-failure candidate ABORTED" in result.stdout.str()


def test_failure_recorder_allows_unrelated_teardown_as_ordinary_red_noise(pytester: pytest.Pytester) -> None:
    output = _install_candidate_recorder(pytester)
    pytester.makepyfile(
        test_sample="""
        import pytest

        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown")

        def test_teardown_only(broken_teardown):
            pass
        """
    )

    result = pytester.runpytest("-q")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert (
        parse_manifest(
            get_profile("cpu-float16"), output.read_text(encoding="utf-8"), current_environment(), source=output
        )
        == {}
    )


def test_failure_recorder_refuses_nodes_fed_by_eager_rng(pytester: pytest.Pytester) -> None:
    output = _install_candidate_recorder(pytester)
    output.write_text("preserve me\n", encoding="utf-8")
    test_path = pytester.path / "tests" / "enhance"
    test_path.mkdir(parents=True)
    (test_path / "test_core.py").write_text(
        """
class TestAddWeighted:
    def test_shape(self):
        raise AssertionError("would be non-reproducible")
""",
        encoding="utf-8",
    )

    # Keep the nested collection isolated from modules imported by the outer full-suite run.
    result = pytester.runpytest_subprocess("tests", "-q")

    assert output.read_text(encoding="utf-8") == "preserve me\n"
    assert "eager RNG" in result.stdout.str()


def test_record_destination_rejects_checked_in_and_symlink_aliases(tmp_path: Path) -> None:
    from testing.half_precision_ci import validate_record_destination

    root = Path(project_conftest.__file__).parent
    checked_in = root / "testing" / "half_precision_xfails" / "cpu_float16.txt"
    alias = tmp_path / "candidate.txt"
    try:
        alias.symlink_to(checked_in)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    destinations = (
        checked_in,
        checked_in.relative_to(root),
        Path("testing/../testing/half_precision_xfails/cpu_float16.txt"),
    )
    for destination in destinations:
        with pytest.raises(ValueError, match="checked-in manifest"):
            validate_record_destination(destination, root)
    with pytest.raises(ValueError, match="checked-in manifest"):
        validate_record_destination(alias, root)


def test_record_destination_rejects_other_git_tracked_files() -> None:
    from testing.half_precision_ci import validate_record_destination

    root = Path(project_conftest.__file__).parent
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("requires Git and a Git checkout")
    with pytest.raises(ValueError, match="Git-tracked"):
        validate_record_destination(root / "TESTING.md", root)


def test_previous_manifest_entries_allow_environment_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from testing.half_precision_ci import ManifestProfile, _previous_manifest_entries

    profile = ManifestProfile("cpu-float16", "float16", tmp_path / "cpu_float16.txt")
    recorded_environment = current_environment()
    entry = ManifestEntry("call", "builtins.AssertionError", "tests/a.py::test_a[cpu-float16]")
    profile.manifest_path.write_text(
        serialize_manifest(profile, [entry], recorded_environment, "pytest tests/"), encoding="utf-8"
    )
    monkeypatch.setattr(
        "testing.half_precision_ci.current_environment",
        lambda: replace(recorded_environment, python="0.0", pytorch="0.0"),
    )

    with pytest.warns(UserWarning, match="compatibility differs"):
        assert _previous_manifest_entries(profile) == {entry.nodeid: entry}


def _run_exact_manifest_case(
    pytester: pytest.Pytester,
    *,
    expected_phase: str = "call",
    expected_exception: str = "builtins.AssertionError",
    test_body: str = "raise AssertionError('known failure')",
    setup_body: str = "pass",
    teardown_body: str = "pass",
    complete: bool = False,
) -> pytest.RunResult:
    nodeid = "test_sample.py::test_known_failure[cpu-float16]"
    _write_profile_manifest(pytester.path, expected_phase, expected_exception, nodeid)
    pytester.makeconftest(
        f"""
        from pathlib import Path

        from testing.half_precision_ci import KnownFailureTracker, get_profile


        def pytest_configure(config):
            tracker = KnownFailureTracker(
                get_profile("cpu-float16"),
                Path(__file__).with_name("cpu_float16.txt"),
                mode={"complete" if complete else "focus"!r},
                selectors=(),
                rootpath=config.rootpath,
            )
            config.pluginmanager.register(tracker, "known-half-precision-failure-tracker")
        """
    )
    pytester.makepyfile(
        test_sample=f"""
        import pytest


        @pytest.fixture(autouse=True)
        def lifecycle_fixture():
            {setup_body}
            yield
            {teardown_body}


        @pytest.mark.parametrize("unused", [None], ids=["cpu-float16"])
        def test_known_failure(unused):
            {test_body}
        """
    )
    return pytester.runpytest("-q")


class TestExactKnownFailureLifecycle:
    @pytest.mark.parametrize("phase", ["setup", "call"])
    def test_accepts_exact_phase_and_identity(self, pytester: pytest.Pytester, phase: str) -> None:
        kwargs = {"setup_body": "raise AssertionError('setup')"} if phase == "setup" else {}

        result = _run_exact_manifest_case(pytester, expected_phase=phase, **kwargs)

        result.assert_outcomes(xfailed=1)
        assert result.ret == pytest.ExitCode.OK

    @pytest.mark.parametrize(
        ("expected_phase", "expected_exception", "test_body", "setup_body", "observed"),
        [
            ("call", "builtins.RuntimeError", "raise NotImplementedError('subclass')", "pass", "NotImplementedError"),
            ("setup", "builtins.AssertionError", "raise AssertionError('call')", "pass", "call"),
            ("call", "builtins.AssertionError", "pytest.skip('runtime skip')", "pass", "skip"),
        ],
    )
    def test_rejects_wrong_exact_outcome_with_diagnostic(
        self,
        pytester: pytest.Pytester,
        expected_phase: str,
        expected_exception: str,
        test_body: str,
        setup_body: str,
        observed: str,
    ) -> None:
        result = _run_exact_manifest_case(
            pytester,
            expected_phase=expected_phase,
            expected_exception=expected_exception,
            test_body=test_body,
            setup_body=setup_body,
        )

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        output = result.stdout.str()
        assert "expected" in output
        assert expected_phase in output
        assert expected_exception in output
        assert observed in output
        assert "test_sample.py::test_known_failure[cpu-float16]" in output

    def test_rejects_teardown_after_matching_call(self, pytester: pytest.Pytester) -> None:
        result = _run_exact_manifest_case(
            pytester, teardown_body="raise RuntimeError('teardown after expected call failure')"
        )

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert "teardown" in result.stdout.str()

    def test_collection_error_has_no_stale_manifest_advice(self, pytester: pytest.Pytester) -> None:
        _write_profile_manifest(
            pytester.path,
            "call",
            "builtins.AssertionError",
            "test_sample.py::test_known_failure[cpu-float16]",
        )
        pytester.makeconftest(
            """
            from pathlib import Path
            from testing.half_precision_ci import KnownFailureTracker, get_profile

            def pytest_configure(config):
                tracker = KnownFailureTracker(
                    get_profile("cpu-float16"), Path(__file__).with_name("cpu_float16.txt"),
                    mode="complete", selectors=(), rootpath=config.rootpath,
                )
                config.pluginmanager.register(tracker, "known-half-precision-failure-tracker")
            """
        )
        pytester.makepyfile(test_sample="raise RuntimeError('import exploded')")

        result = pytester.runpytest("-q")

        assert result.ret == pytest.ExitCode.INTERRUPTED
        output = result.stdout.str()
        assert "import exploded" in output
        assert "remove or update" not in output
        assert "not collected" not in output

    def test_collect_only_does_not_validate_runtime_outcomes(self, pytester: pytest.Pytester) -> None:
        nodeid = "test_sample.py::test_known_failure[cpu-float16]"
        _write_profile_manifest(pytester.path, "call", "builtins.AssertionError", nodeid)
        pytester.makeconftest(
            """
            from pathlib import Path
            from testing.half_precision_ci import KnownFailureTracker, get_profile

            def pytest_configure(config):
                tracker = KnownFailureTracker(
                    get_profile("cpu-float16"), Path(__file__).with_name("cpu_float16.txt"),
                    mode="complete", selectors=(), rootpath=config.rootpath,
                )
                config.pluginmanager.register(tracker, "known-half-precision-failure-tracker")
            """
        )
        pytester.makepyfile(
            test_sample="""
            import pytest

            @pytest.mark.parametrize("unused", [None], ids=["cpu-float16"])
            def test_known_failure(unused):
                raise AssertionError("known failure")
            """
        )

        result = pytester.runpytest("--collect-only", "-q")

        assert result.ret == pytest.ExitCode.OK
        assert "known half-precision failure" not in result.stdout.str()


class _RecordingConfig:
    def __init__(self, option_name: str, option_value: object, rootpath: Path) -> None:
        options = {
            "keyword": "",
            "markexpr": "",
            "deselect": [],
            "maxfail": 0,
            "lf": False,
            "failedfirst": False,
            "collectonly": False,
            "numprocesses": None,
            "plugins": [],
        }
        options[option_name] = option_value
        self.option = SimpleNamespace(**options)
        self.args = ["tests"]
        self.rootpath = rootpath
        self.invocation_params = SimpleNamespace(args=())


class _ProfileConfig:
    def __init__(
        self,
        *,
        profile: str | None = "cpu-float16",
        xfail: bool = True,
        verify: bool = False,
        record: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        args: tuple[str, ...] = (),
    ) -> None:
        self.option = SimpleNamespace(device=device, dtype=dtype)
        self.invocation_params = SimpleNamespace(args=args)
        self._options = {
            "--known-failure-profile": profile,
            "--xfail-known-failures": xfail,
            "--verify-known-failures": verify,
            "--record-known-failures": record,
        }

    def getoption(self, option: str) -> object:
        return self._options[option]


class TestProfileConfiguration:
    def test_applies_profile_before_parametrization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KORNIA_TEST_DEVICE", raising=False)
        monkeypatch.delenv("KORNIA_TEST_DTYPE", raising=False)
        config = _ProfileConfig()

        profile = project_conftest._configure_known_failure_profile(config)

        assert profile == get_profile("cpu-float16")
        assert config.option.device == "cpu"
        assert config.option.dtype == "float16"

    @pytest.mark.parametrize(
        ("args", "environment", "match"),
        [
            (("--device=cuda",), {}, "device.*cuda.*cpu"),
            (("--dtype", "float64"), {}, "dtype.*float64.*float16"),
            ((), {"KORNIA_TEST_DTYPE": "bfloat16"}, "dtype.*bfloat16.*float16"),
        ],
    )
    def test_rejects_explicit_profile_conflicts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        args: tuple[str, ...],
        environment: dict[str, str],
        match: str,
    ) -> None:
        monkeypatch.delenv("KORNIA_TEST_DEVICE", raising=False)
        monkeypatch.delenv("KORNIA_TEST_DTYPE", raising=False)
        for key, value in environment.items():
            monkeypatch.setenv(key, value)
        config = _ProfileConfig(args=args, device="cuda" if "device" in match else "cpu", dtype="float64")
        if environment:
            config.option.dtype = environment["KORNIA_TEST_DTYPE"]

        with pytest.raises(pytest.UsageError, match=match):
            project_conftest._configure_known_failure_profile(config)

    @pytest.mark.parametrize(("verify", "record"), [(True, None), (False, "candidate.txt")])
    def test_complete_modes_require_explicit_profile(self, verify: bool, record: str | None) -> None:
        config = _ProfileConfig(profile=None, xfail=False, verify=verify, record=record)

        with pytest.raises(pytest.UsageError, match="requires --known-failure-profile"):
            project_conftest._configure_known_failure_profile(config)

    def test_unprofiled_xfail_keeps_legacy_mps_configuration(self) -> None:
        config = _ProfileConfig(profile=None, xfail=True, dtype="float32")

        assert project_conftest._configure_known_failure_profile(config) is None
        assert config.option.device == "cpu"
        assert config.option.dtype == "float32"


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("keyword", "selected"),
        ("markexpr", "unit"),
        ("deselect", ["tests/a.py::test_a"]),
        ("maxfail", 1),
        ("lf", True),
        ("failedfirst", True),
        ("collectonly", True),
        ("numprocesses", 2),
        ("plugins", ["no:cacheprovider"]),
    ],
)
def test_record_mode_rejects_partial_run_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option_name: str, option_value: object
) -> None:
    monkeypatch.delenv("KORNIA_TEST_OPTIMIZER", raising=False)
    config = _RecordingConfig(option_name, option_value, tmp_path)

    with pytest.raises(pytest.UsageError, match="does not allow partial-selection options"):
        project_conftest._validate_complete_known_failure_run(config, "--record-known-failures")


def test_record_mode_rejects_minus_p_when_cacheprovider_is_disabled(tmp_path: Path) -> None:
    """`-p no:cacheprovider` removes the --lf/--ff options entirely; the guard must not need them."""
    option = SimpleNamespace(
        keyword="", markexpr="", deselect=[], maxfail=0, collectonly=False, plugins=["no:cacheprovider"]
    )
    config = SimpleNamespace(
        option=option, args=["tests"], rootpath=tmp_path, invocation_params=SimpleNamespace(args=())
    )

    with pytest.raises(pytest.UsageError, match="does not allow partial-selection options; remove: -p"):
        project_conftest._validate_complete_known_failure_run(config, "--record-known-failures")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_recorded_baseline_manifest(dtype: str) -> None:
    profile = get_profile(f"cpu-{dtype}")
    environment = replace(current_environment(), python="other", pytorch="other", os="Other", arch="other")
    with pytest.warns(UserWarning, match="compatibility differs"):
        entries = parse_manifest(
            profile,
            profile.manifest_path.read_text(encoding="utf-8"),
            environment,
            source=profile.manifest_path,
            enforce_compatibility=False,
        )

    assert entries
    assert all(entry.phase in {"setup", "call"} for entry in entries.values())
    assert all("." in entry.exception for entry in entries.values())


def test_reusable_test_workflow_exposes_complete_profile_input() -> None:
    root = Path(project_conftest.__file__).parent
    workflow = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    assert "known-failure-profile:" in workflow
    assert "KNOWN_FAILURE_PROFILE: ${{ inputs.known-failure-profile }}" in workflow
    assert "--verify-known-failures" in workflow
    assert '"--known-failure-profile=$KNOWN_FAILURE_PROFILE"' in workflow


def test_reusable_test_workflow_accepts_only_complete_recordings() -> None:
    root = Path(project_conftest.__file__).parent
    workflow = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    assert 'pipeline_status=("${PIPESTATUS[@]}")' in workflow
    assert '[[ "$tee_status" -eq 0 && "$test_status" -le 1 && -s "$candidate_path" ]]' in workflow
    assert 'grep -Fq "known-failure candidate complete for $KNOWN_FAILURE_PROFILE:"' in workflow
    assert "record mode finished without a complete candidate receipt" in workflow


def _upstream_probe_jobs() -> dict[str, str]:
    """Split upstream_probe.yml into its top-level job bodies."""
    root = Path(project_conftest.__file__).parent
    workflow = (root / ".github" / "workflows" / "upstream_probe.yml").read_text(encoding="utf-8")
    jobs: dict[str, str] = {}
    name = None
    for line in workflow.splitlines():
        match = re.fullmatch(r"  (\w[\w-]*):", line)
        if match:
            name = match.group(1)
            jobs[name] = ""
        elif name is not None:
            jobs[name] += line + "\n"
    return jobs


def test_upstream_probe_jobs_all_use_floating_stable_torch() -> None:
    jobs = {name: body for name, body in _upstream_probe_jobs().items() if "workflows/tests.yml" in body}

    assert set(jobs) == {"cpu-float32", "cpu-float64", "cpu-float16", "cpu-bfloat16", "python314"}
    for name, body in jobs.items():
        assert "\n      torch-channel: stable\n" in body, name


def test_upstream_probe_floor_is_consistent_across_jobs() -> None:
    # The floor has to be repeated per job because a reusable workflow's `with:` cannot read a
    # job-level `env:`; a bump that misses one job would silently stop probing that surface.
    floors = {
        name: re.search(r"\n      pytorch-version: '\[\"([^\"]+)\"\]'", body).group(1)
        for name, body in _upstream_probe_jobs().items()
        if "workflows/tests.yml" in body
    }

    assert len(set(floors.values())) == 1, floors


def test_upstream_probe_record_jobs_match_the_blocking_python() -> None:
    # A candidate manifest carries the recording Python in its header and the blocking
    # half-precision job refuses a mismatch, so the record-mode probes must use the same
    # Python that pr_test_cpu.yml's half-precision job uses.
    root = Path(project_conftest.__file__).parent
    blocking = (root / ".github" / "workflows" / "pr_test_cpu.yml").read_text(encoding="utf-8")
    blocking_python = re.search(
        r"  half-precision:.*?\n      python-version: '\[\"([^\"]+)\"\]'", blocking, re.DOTALL
    ).group(1)

    for name, body in _upstream_probe_jobs().items():
        if "known-failure-mode: record" not in body:
            continue
        assert f"\n      python-version: '[\"{blocking_python}\"]'\n" in body, name


def test_reusable_test_workflow_names_candidate_by_resolved_torch() -> None:
    # Under torch-channel: stable the matrix value is only a floor, so naming the artifact after
    # it would make every future probe upload collide on the same stale name.
    root = Path(project_conftest.__file__).parent
    workflow = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    assert 'echo "resolved=$resolved" >> "$GITHUB_OUTPUT"' in workflow
    assert (
        "name: candidate-manifest-${{ inputs.known-failure-profile }}-${{ matrix.python-version }}"
        "-torch${{ steps.torch-version.outputs.resolved }}"
    ) in workflow
    assert "-${{ matrix.pytorch-version }}\n" not in workflow.partition("Upload candidate manifest")[2]


def test_pinned_runner_images_have_availability_guards() -> None:
    root = Path(project_conftest.__file__).parent
    pr_workflow = (root / ".github" / "workflows" / "pr_test_cpu.yml").read_text(encoding="utf-8")
    scheduled_workflow = (root / ".github" / "workflows" / "scheduled_test_cpu.yml").read_text(encoding="utf-8")

    _, separator, tests_mps_tail = pr_workflow.partition("  tests-mps:\n")
    assert separator, "missing tests-mps job"
    tests_mps = re.split(r"\n(?=  [\w-]+:\n)", tests_mps_tail, maxsplit=1)[0]
    assert "\n      os: macos-15\n" in tests_mps
    assert "          - os: windows-2022\n" in pr_workflow
    assert "          - os: windows-2022\n" in scheduled_workflow

    for label in ("macos-15", "windows-2022"):
        _, separator, tail = scheduled_workflow.partition(f"  pinned-runner-{label}:\n")
        assert separator, f"missing scheduled availability guard for {label}"
        guard = re.split(r"\n(?=  [\w-]+:\n)", tail, maxsplit=1)[0]

        assert f"    name: Pinned runner availability ({label})\n" in guard
        assert (
            "    if: >-\n"
            "      (github.event_name == 'schedule' || github.event_name == 'workflow_dispatch') &&\n"
            "      github.repository == 'kornia/kornia' &&\n"
            "      startsWith(github.workflow_ref, 'kornia/kornia/.github/workflows/scheduled_test_cpu.yml@')\n"
        ) in guard
        assert f"    runs-on: {label}\n" in guard
        assert "    timeout-minutes: 5\n" in guard
        assert "    needs:" not in guard


@pytest.mark.parametrize("workflow_name", ["pr_test_cpu.yml", "scheduled_test_cpu.yml"])
def test_cpu_half_workflow_uses_complete_profile(workflow_name: str) -> None:
    root = Path(project_conftest.__file__).parent
    workflow = (root / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")

    assert "known-failure-profile: cpu-${{ matrix.pytorch-dtype }}" in workflow
