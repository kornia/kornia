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

"""Support for the conformance tests of #5819: kornia's ONNX and torch.compile promises, checked with the survey's
own cases.

Two promises share this machinery, ``ONNX`` and ``COMPILE``, defined in ``testing/conformance_promise.py``: the
promised cases of each are the records of the committed survey snapshot (``docs/source/_data/export_support.json``)
whose status for it (field ``onnx`` or ``compile``) is ``ok`` and whose package is in the stable core of
``docs/source/get-started/stability.rst``, minus everything built with pretrained weights (registry tag
``pretrained``) and the promise's listed exclusions. Each is run by case name through the survey's own code, so the
tests and the survey page cannot drift apart:

- ``tests/onnx/test_conformance.py`` exports through ``docs/export_support/harness.py`` (``run_onnx_case``);
- ``tests/test_inductor_conformance.py`` compiles through ``docs/export_support/probe_compile.py``
  (``run_compile_case``: ``torch.compile(..., fullgraph=True)``, default inductor backend).

Each run is opt-in: ``--onnx-conformance`` or ``--compile-conformance`` with ``<package>[,<package>...]`` or ``all``
(env ``KORNIA_TEST_ONNX_CONFORMANCE`` / ``KORNIA_TEST_COMPILE_CONFORMANCE``); without it the root ``conftest.py``
deselects every test with that promise's marker. Known failures are strict xfails listed per promise and torch release
in ``testing/known_failure_xfails/<onnx|compile>_conformance_torch<major>.<minor>.txt``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import logging
import pickle
import signal
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

import pytest
import torch

# The promise itself (what is promised, left out, known to fail) is testing/conformance_promise.py, which the docs build
# imports without pytest; this module runs it as pytest tests.
from testing.conformance_promise import (
    EXPORT_SUPPORT,
    REGISTRIES,
    STABLE_CORE,
    Promise,
    exclusion_problems,
    known_failures,
    manifest_path,
    pretrained_cases,
    promised_cases,
    promised_digest,
)
from testing.known_failures import _EXCEPTION_TYPES, KnownFailureTracker

# The registries that hold each package's promised cases. ``cases_models.py`` is never imported: it builds the
# pretrained models of ``kornia.models``, none of which is promised, and downloads their weights on a cold cache.
# A promised case outside its package's registries fails test_promised_cases_are_registered.
REGISTRIES_OF = {
    "geometry": ("cases_geomA", "cases_geomB"),
    "augmentation": ("cases_aug",),
    "color": ("cases_misc",),
    "filters": ("cases_misc",),
    "enhance": ("cases_misc",),
    "losses": ("cases_misc",),
    "metrics": ("cases_misc",),
    "morphology": ("cases_misc",),
    "io": (),
    "image": ("cases_misc",),
    "core": (),
    "feature": ("cases_feature",),
}

# Legacy (TorchScript) export statuses that pass: the export worked and matched, or it failed loudly.
LEGACY_PASSES = ("ok", "export-fail", "checker-fail", "ort-load-fail", "ort-run-fail")

# The exception a per-case test raises for a status that does not pass, so that a manifest entry pins the kind of
# failure: a known failure that changes kind (a graph break that turns into a wrong value, an eager regression of a
# pinned case) then fails as an unexpected exception instead of staying a strict xfail.
STATUS_EXCEPTIONS: dict[str, type[Exception]] = {
    # the export or compile ran and the result differs from eager: value, dtype, shape or output count
    "mismatch": AssertionError,
    # the export, the ONNX checker, onnxruntime or the compile failed (a graph break, a backend error)
    "fail": RuntimeError,
    "export-fail": RuntimeError,
    "checker-fail": RuntimeError,
    "ort-load-fail": RuntimeError,
    "ort-run-fail": RuntimeError,
    # the eager reference itself failed, or returned no tensor
    "eager-fail": ValueError,
    "no-tensor-output": ValueError,
}

# torch's process-wide module hooks that a failed or (on torch 2.5) a successful non-strict ``torch.export`` trace can
# leave registered; a later ``torch.compile`` of a module with buffers then fails. The runners remove what they added.
REGISTRATION_HOOKS = (
    "_global_buffer_registration_hooks",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)


def parse_selection(promise: Promise, value: str | None) -> tuple[str, ...]:
    """Return the packages a ``--<promise>-conformance`` value selects; ``()`` when it is unset (``None`` or ``""``).

    Args:
        promise: the promise whose option carries the value.
        value: ``all``, or comma-separated package names with or without the ``kornia.`` prefix.

    Returns:
        The selected packages, in ``STABLE_CORE`` order.

    Raises:
        ValueError: when a name is empty (a blank value, a stray comma) or not a stable-core package.
    """
    if value is None or value == "":
        return ()
    names = [v.strip().removeprefix("kornia.") for v in value.split(",")]
    if not all(names):
        raise ValueError(f"{promise.option}: empty package name in {value!r}; give packages or all")
    unknown = sorted(set(names) - set(STABLE_CORE) - {"all"})
    if unknown:
        raise ValueError(f"{promise.option}: unknown package(s) {unknown}; choose from {', '.join(STABLE_CORE)} or all")
    if "all" in names:
        return STABLE_CORE
    return tuple(p for p in STABLE_CORE if p in names)


def selection(config: pytest.Config, promise: Promise) -> tuple[str, ...]:
    """The packages the session selects for ``promise``."""
    return parse_selection(promise, config.getoption(promise.option))


def selected_cases(promise: Promise, packages: Sequence[str]) -> list[str]:
    """Return the promised case names of ``packages``, in snapshot order."""
    return [name for name, package in promised_cases(promise).items() if package in packages]


def parametrize_cases(metafunc: pytest.Metafunc, promise: Promise) -> None:
    """Parametrize a per-case test (argument ``case_name``) with the promised cases the session selects."""
    if "case_name" not in metafunc.fixturenames:
        return
    names = selected_cases(promise, selection(metafunc.config, promise))
    # Without a selection the root conftest deselects the placeholder; with one that has no promised case it skips.
    metafunc.parametrize("case_name", names or [pytest.param(None, id="unselected")])


@contextlib.contextmanager
def _no_downloads() -> Iterator[None]:
    """Make a weights download raise instead: no case the registries build may reach the network here."""
    from testing.doctest_downloads import install_download_guard

    undo: list[tuple[Any, str, Any]] = []

    def set_attribute(obj: Any, name: str, value: Any) -> None:
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def refuse(url: str) -> None:
        raise RuntimeError(f"weights are not in the cache (the conformance test does not download): {url}")

    try:
        install_download_guard(set_attribute, refuse)
        yield
    finally:
        for obj, name, value in reversed(undo):
            setattr(obj, name, value)


@functools.cache
def import_survey_module(name: str) -> ModuleType:
    """Import ``docs/export_support/<name>.py`` without letting the import touch the process.

    The registries seed the global RNG and build their inputs (and the pretrained models of cases that are not
    promised) at import, and print the traceback of a model they cannot build: the RNG state is restored, downloads
    refused and the output dropped.
    """
    path = str(EXPORT_SUPPORT)
    sys.path.insert(0, path)
    try:
        sink = io.StringIO()
        with (
            torch.random.fork_rng(devices=[]),
            _no_downloads(),
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            return importlib.import_module(name)
    finally:
        sys.path.remove(path)


def registries_of(packages: Sequence[str]) -> tuple[str, ...]:
    """The registries holding the promised cases of ``packages``, in ``REGISTRIES`` order."""
    needed = {registry for package in packages for registry in REGISTRIES_OF[package]}
    return tuple(registry for registry in REGISTRIES if registry in needed)


@functools.cache
def registry_cases(registries: tuple[str, ...] = REGISTRIES) -> dict[str, dict[str, Any]]:
    """Every case of ``registries`` by name; a name registered twice is an error."""
    cases: dict[str, dict[str, Any]] = {}
    for registry in registries:
        for case in import_survey_module(registry).CASES:
            if case["name"] in cases:
                raise ValueError(f"case {case['name']!r} is registered twice ({registry})")
            cases[case["name"]] = case
    return cases


def promised_case(promise: Promise, case_name: str | None) -> dict[str, Any]:
    """The registry case of a promised name; skips the ``unselected`` placeholder, fails when nothing runs it."""
    if case_name is None:
        pytest.skip("the selected packages have no promised case")
    case = registry_cases(registries_of([promised_cases(promise)[case_name]])).get(case_name)
    if case is None or case["skip"] or not case["check"]:
        pytest.fail(f"{case_name!r} is promised but no registry case runs it; see test_promised_cases_are_registered")
    return case


def registration_problems(promise: Promise, packages: Sequence[str]) -> list[str]:
    """Promised cases of ``packages`` that no registry case runs and compares with eager (the drift guard)."""
    registries = registries_of(packages)
    cases = registry_cases(registries)
    problems = []
    for name in selected_cases(promise, packages):
        case = cases.get(name)
        if case is None:
            problems.append(f"{name}: no case of that name in {', '.join(registries) or 'no registry'}")
        elif case["skip"]:
            problems.append(f"{name}: skipped ({case['skip']})")
        elif not case["check"]:
            problems.append(f"{name}: check=False, the output is not compared with eager")
    # The promise leaves out pretrained cases by reading the registry sources; the imported registries must agree.
    tagged = {name for name, case in cases.items() if "pretrained" in case["tags"]}
    if disagree := tagged ^ (pretrained_cases() & cases.keys()):
        problems.append(f"registry tags and the static pretrained scan disagree on {sorted(disagree)}")
    return problems


def pin_problems(promise: Promise, count: int, digest: str) -> list[str]:
    """How the promised set differs from its pin (``count``, ``digest``), keeps a stale exclusion, or breaks a rule."""
    # The promise is read from the snapshot, so a regeneration that measured a case as failing would quietly drop it.
    # Pinning the names makes a change of the promised set a reviewed change of these two constants.
    promised = promised_cases(promise)
    problems = []
    if (len(promised), promised_digest(promised)) != (count, digest):
        problems.append(
            "the promised set changed; review the snapshot diff, then update PROMISED_COUNT and PROMISED_DIGEST"
            f" (now {len(promised)}, {promised_digest(promised)})"
        )
    problems += exclusion_problems(promise)
    if outside := [name for name in promised if name.startswith(("models.", "contrib."))]:
        problems.append(f"promised cases outside the stable core: {outside}")
    if pretrained := sorted(promised.keys() & pretrained_cases()):
        problems.append(f"promised cases built with pretrained weights: {pretrained}")
    # The root conftest deselects or skips tests whose id contains these words, which would drop the case silently.
    if words := [name for name in promised if any(word in name.lower() for word in ("dynamo", "compile", "autocast"))]:
        problems.append(f"promised case names the root conftest deselects or skips: {words}")
    return problems


def registration_hooks() -> dict[str, frozenset[Any]]:
    """The handles registered in each of torch's process-wide module registration hooks that this torch has."""
    from torch.nn.modules import module

    return {
        name: frozenset(hooks) for name in REGISTRATION_HOOKS if isinstance(hooks := getattr(module, name, None), dict)
    }


@contextlib.contextmanager
def _restore_registration_hooks() -> Iterator[None]:
    from torch.nn.modules import module

    before = registration_hooks()
    try:
        yield
    finally:
        for name, handles in before.items():
            hooks = getattr(module, name)
            for handle in [handle for handle in hooks if handle not in handles]:
                del hooks[handle]


def _compile_configs() -> tuple[Any, ...]:
    """The dynamo and inductor config modules, or none on a torch whose Dynamo cannot be imported."""
    try:
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config
    except ImportError:
        return ()
    return dynamo_config, inductor_config


def _same_value(a: Any, b: Any) -> bool:
    """Whether two copies of a torch config value are equal; a value need not define ``==``, so compare pickles."""
    try:
        if a is b or bool(a == b):
            return True
    except Exception:  # noqa: BLE001, S110 -- an ``==`` that cannot answer falls through to the pickles
        pass
    try:
        return pickle.dumps(a) == pickle.dumps(b)
    except Exception:  # noqa: BLE001 -- a value pickle rejects is equal only to itself
        return False


def changed_config_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The keys whose values differ between two ``get_config_copy()`` results of one torch config module."""
    return [key for key in before.keys() | after.keys() if not _same_value(before.get(key), after.get(key))]


@contextlib.contextmanager
def _restore_compile_configs() -> Iterator[None]:
    """Put back every dynamo and inductor config key whose value changed inside the block.

    ``get_config_copy`` and ``setattr`` are the config API every supported torch has (torch 2.5's ``save_config``
    fails on its own Dynamo config).
    """
    saved = [(config, config.get_config_copy()) for config in _compile_configs()]
    try:
        yield
    finally:
        for config, before in saved:
            for key in changed_config_keys(before, config.get_config_copy()):
                setattr(config, key, before[key])


def run_onnx_case(case: dict[str, Any], dynamo: bool) -> dict[str, Any]:
    """Run one case through the ONNX survey harness, leaving the global RNG and torch's module hooks as they were."""
    harness = import_survey_module("harness")
    with torch.random.fork_rng(devices=[]), _restore_registration_hooks():
        return harness.run_case(case, dynamo=dynamo)


def run_compile_case(case: dict[str, Any]) -> dict[str, Any]:
    """Compile one case with ``fullgraph=True`` through the survey's compile probe and compare it with eager.

    The probe resets Dynamo before and after the case; the global RNG, torch's module hooks and the dynamo and
    inductor configs are left as they were.
    """
    probe = import_survey_module("probe_compile")
    with torch.random.fork_rng(devices=[]), _restore_registration_hooks(), _restore_compile_configs():
        return probe.run_case(case, export=False, fullgraph=True)


def raise_for_status(record: dict[str, Any], passes: Sequence[str], step: str = "") -> None:
    """Raise the ``STATUS_EXCEPTIONS`` type of a record whose status is not in ``passes``, with ``describe``'s message.

    Args:
        record: a survey record.
        passes: the statuses that pass.
        step: as for ``describe``: ``""`` for a harness record, ``"compile"`` for a compile probe record.
    """
    status = record[step or "status"]
    if status not in passes:
        raise STATUS_EXCEPTIONS.get(status, AssertionError)(describe(record, step))


def describe(record: dict[str, Any], step: str = "") -> str:
    """A failure message naming the status, the error and the kornia frames involved.

    Args:
        record: a survey record.
        step: ``""`` for a harness record (``status``, ``error``, ...), ``"compile"`` for a compile probe record
            (``compile``, ``compile_error``, ...).
    """
    prefix = f"{step}_" if step else ""
    # the compile probe records an eager failure under ``error``, before any step ran
    lines = [f"{record[step or 'status']}: {record.get(prefix + 'error') or record.get('error', '')}"]
    lines += [f"  at {frame}" for frame in record.get(prefix + "where") or ()]
    if record.get(prefix + "error_full"):
        lines.append(record[prefix + "error_full"][-1500:])
    return "\n".join(lines)


def process_state() -> dict[str, Any]:
    """The process-wide state a survey run must leave alone, for ``state_changes``."""
    configs = _compile_configs()  # first: importing Dynamo the first time adds sympy's warning filter
    state: dict[str, Any] = {
        "threads": torch.get_num_threads(),
        "logging": logging.root.manager.disable,
        "filters": list(warnings.filters),
        "rng": torch.random.get_rng_state(),
        "hooks": registration_hooks(),
    }
    if hasattr(signal, "SIGALRM"):
        state["alarm handler"] = signal.getsignal(signal.SIGALRM)
    for config in configs:
        state[config.__name__] = config.get_config_copy()
    return state


def state_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """What changed between two ``process_state`` snapshots.

    A torch.export trace or a Dynamo frame conversion may add torch's own warning filters (torch 2.5 adds one for
    module "torch.distributed"); the harness used to add a blanket "ignore everything" one. So a new filter passes only
    when it names a torch module.
    """
    changes = []
    for key in before.keys() | after.keys():
        old, new = before.get(key), after.get(key)
        if key == "rng":
            if not torch.equal(old, new):
                changes.append("the global CPU RNG state")
        elif key == "filters":
            added = [f for f in new if f not in old]
            if not all(f[3] is not None and f[3].pattern.startswith("torch") for f in added):
                changes.append(f"warning filters added: {added}")
        elif key.startswith("torch."):  # a dynamo or inductor config
            changes += [f"{key}.{name}" for name in sorted(changed_config_keys(old, new))]
        elif isinstance(old, dict) and isinstance(new, dict):
            changes += [f"{key}.{name}" for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)]
        elif old != new:
            changes.append(f"{key}: {old!r} -> {new!r}"[:300])
    return sorted(changes)


def load_known_failures(promise: Promise, path: Path) -> dict[str, type[BaseException]]:
    """Read a promise's manifest (``conformance_promise.known_failures``): node ID -> the exception type it names."""
    failures: dict[str, type[BaseException]] = {}
    for entry in known_failures(promise, path):
        if entry.exception not in _EXCEPTION_TYPES:
            raise ValueError(f"{path}:{entry.line}: unknown exception type: {entry.exception}")
        failures[entry.nodeid] = _EXCEPTION_TYPES[entry.exception]
    return failures


class KnownFailurePlugin:
    """Apply a promise's manifest for this torch to the final selection (after ``-k``, ``-m``, ``--deselect``)."""

    def __init__(self, promise: Promise) -> None:
        self.promise = promise

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[Any]) -> None:
        try:
            tracker = mark_known_failures(self.promise, items)
        except (OSError, ValueError) as error:
            raise pytest.UsageError(str(error)) from error
        if tracker is not None:
            config.pluginmanager.register(tracker, f"{self.promise.field}-conformance-known-failure-tracker")


def mark_known_failures(promise: Promise, items: Sequence[Any], path: Path | None = None) -> KnownFailureTracker | None:
    """Strictly xfail the collected entries of a promise's manifest for the running torch; ``None`` when it has none.

    Entries of packages the run did not select are not collected and not checked. A collected entry that passes
    (strict XPASS), skips, or fails with another exception type fails the session through the tracker.
    """
    path = path or manifest_path(promise)
    if not path.exists():
        return None
    failures = load_known_failures(promise, path)
    collected = {item.nodeid for item in items}
    selected = {nodeid: exception for nodeid, exception in failures.items() if nodeid in collected}
    reason = f"Known {promise.name} conformance failure recorded in {path.name}; see its tracking issue there"
    for item in items:
        if exception := selected.get(item.nodeid):
            item.add_marker(pytest.mark.xfail(reason=reason, raises=exception, strict=True))
    return KnownFailureTracker(selected, path, reason)
