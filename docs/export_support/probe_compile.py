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

"""torch.export / torch.compile probe over the survey cases (same registry files as the ONNX survey).

Usage: python probe_compile.py <cases_module> <out.json> [names...]

Per case records:
  export:  ok | fail | eager-fail | skipped         (torch.export.export, non-strict default)
  compile: ok | ok-breaks:N | ok-unverified | mismatch | fail | eager-fail | skipped
           (torch.compile, default inductor backend: ok only for one graph, fullgraph=True; a case fullgraph=True
           refuses is compiled with fullgraph=False, N = the graph breaks torch._dynamo.explain counts)
plus the target's importable qualified name when it has one.

``run_case(c, export=False, fullgraph=True)`` is the compile step alone with ``torch.compile(..., fullgraph=True)``,
where a graph break fails the case: the check ``tests/test_inductor_conformance.py`` runs on the promised cases.
Importing this module leaves the process alone (``main`` installs the per-case timeout and the survey's settings).
"""

from __future__ import annotations

import copy
import importlib
import json
import signal
import sys
import time
from typing import Any

import torch
import torch._dynamo
from harness import (
    SEED,
    _kornia_frames,
    close,
    configure_survey_process,
    err_full,
    err_str,
    make_wrapper,
    prepare_resume,
    shape_and_finite,
)
from torch import nn

TIMEOUT = 420


class Timeout(Exception):
    pass


class Poisoned(Exception):
    """A previous case left the dispatcher unusable (PythonDispatcherTLS not set): restart the worker."""


POISON_EXIT = 3


def _alarm(signum, frame):
    raise Timeout(f"case exceeded {TIMEOUT}s")


_HAS_ALARM = hasattr(signal, "SIGALRM")  # POSIX only; on Windows the probes simply run without a timeout


def _set_alarm(seconds: int) -> None:
    # only under main's handler: another process (a pytest session) keeps its own SIGALRM, and without a handler
    # the signal would end the process
    if _HAS_ALARM and signal.getsignal(signal.SIGALRM) is _alarm:
        signal.alarm(seconds)


def qualname(target: Any, method: str | None) -> str | None:
    obj = target
    if isinstance(target, nn.Module):
        obj = type(target)
    if method and hasattr(obj, method):
        obj = getattr(obj, method)
    mod = getattr(obj, "__module__", None) or ""
    qn = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not mod.startswith("kornia") or not qn or "<lambda>" in qn or "<locals>" in qn:
        return None
    return f"{mod}.{qn}"


def _compare(ref: list[torch.Tensor], out: list[torch.Tensor], c: dict[str, Any]) -> str | None:
    if len(out) != len(ref):
        return f"{len(out)} outputs vs {len(ref)}"
    bad = []
    for i, (a, b) in enumerate(zip(ref, out)):
        ok, why = close(a, b, c["atol"], c["rtol"])
        if not ok:
            bad.append(f"out{i}: {why}")
    return "; ".join(bad)[:400] if bad else None


def _reference(c: dict[str, Any]) -> tuple[nn.Module, list[torch.Tensor], list[torch.Tensor]]:
    """The case's wrapper, a copy of its inputs, and its eager outputs.

    The eager call runs on a copy of the wrapper, and the export and compile steps take their own copies, so none
    reads the state another left on a module (an augmentation's params, a crop's cached points, an export's fake
    tensors): a compiled graph that skips computing such state cannot pass by reading what eager left there, and the
    case, which later tests reuse, stays as it was built.
    """
    target = c["target"]
    if isinstance(target, nn.Module):
        target.eval()
    wrap = make_wrapper(target, c["kwargs"], len(c["inputs"]), c["method"])
    inputs = [x.detach().clone() for x in c["inputs"]]
    torch.manual_seed(SEED)
    with torch.no_grad():
        ref = list(copy.deepcopy(wrap)(*copy.deepcopy(inputs)))
    return wrap, inputs, ref


def export_step(wrap: nn.Module, inputs: list[torch.Tensor]) -> dict[str, Any]:
    """``torch.export.export`` of the wrapper: the ``export`` fields of a record."""
    rec: dict[str, Any] = {}
    _set_alarm(TIMEOUT)
    try:
        torch.manual_seed(SEED)
        with torch.no_grad():
            torch.export.export(copy.deepcopy(wrap), tuple(copy.deepcopy(inputs)))
        rec["export"] = "ok"
    except Exception as e:  # not BaseException: KeyboardInterrupt and a test's outcome (pytest-timeout) end the run
        rec.update(export="fail", export_error=err_str(e), export_where=_kornia_frames(e))
    finally:
        _set_alarm(0)
    return rec


def _compiled_outputs(wrap: nn.Module, inputs: list[torch.Tensor], fullgraph: bool) -> list[torch.Tensor]:
    compiled = torch.compile(copy.deepcopy(wrap), fullgraph=fullgraph)
    torch.manual_seed(SEED)
    with torch.no_grad():
        return list(compiled(*copy.deepcopy(inputs)))


def compile_step(
    c: dict[str, Any], wrap: nn.Module, inputs: list[torch.Tensor], ref: list[torch.Tensor], fullgraph: bool = False
) -> dict[str, Any]:
    """``torch.compile`` (default inductor backend) of the wrapper, compared with ``ref``: the ``compile`` fields.

    A case is ``ok`` only when ``torch.compile(..., fullgraph=True)`` captures it in one graph and the result matches
    eager: ``torch._dynamo.explain`` can count no graph break for a call Dynamo runs partly or wholly in Python (#6194).
    With ``fullgraph=True`` (the conformance test) that compile is the whole step, and a graph break fails the case.
    The survey (``fullgraph=False``) goes on for a case ``fullgraph=True`` refuses: explain counts its graph breaks
    and a default ``torch.compile`` is compared with eager, ``ok-breaks:N`` when it matches (``N`` may be 0, the
    refusal then being the recorded break reason). Dynamo is reset before and after each compile.
    """
    rec: dict[str, Any] = {"graph_breaks": 0}
    t1 = time.time()
    _set_alarm(TIMEOUT)
    try:
        torch._dynamo.reset()
        refusal: Exception | None = None
        try:
            out = _compiled_outputs(wrap, inputs, fullgraph=True)
        except Timeout:
            raise
        except Exception as e:  # not BaseException: a test's outcome (pytest-timeout) ends the run
            if fullgraph:
                raise
            refusal = e
        # outside the except block, so that a failure below is not reported as the refusal it would be chained to
        if refusal is not None:
            torch._dynamo.reset()
            torch.manual_seed(SEED)
            with torch.no_grad():
                ex = torch._dynamo.explain(copy.deepcopy(wrap))(*copy.deepcopy(inputs))
            # explain counts -1 when Dynamo captures no graph at all and runs the whole call in Python
            rec["graph_breaks"] = max(int(ex.graph_break_count), 0)
            reasons = sorted({str(r.reason)[:160] for r in ex.break_reasons})[:5]
            rec["break_reasons"] = reasons or [f"fullgraph=True: {err_str(refusal)}"[:160]]
            torch._dynamo.reset()
            out = _compiled_outputs(wrap, inputs, fullgraph=False)
        if not c["check"]:
            why = shape_and_finite(ref, out)
            rec["compile"] = "ok-unverified" if why is None else "mismatch"
        else:
            why = _compare(ref, out, c)
            ok = "ok" if refusal is None else f"ok-breaks:{rec['graph_breaks']}"
            rec["compile"] = ok if why is None else "mismatch"
        if why:
            rec["compile_error"] = why
    except Exception as e:  # the survey's own Timeout is one; a pytest outcome or KeyboardInterrupt is not
        rec.update(
            compile="fail", compile_error=err_str(e), compile_error_full=err_full(e), compile_where=_kornia_frames(e)
        )
    finally:
        _set_alarm(0)
        torch._dynamo.reset()
    rec["compile_s"] = round(time.time() - t1, 1)
    return rec


def run_case(c: dict[str, Any], export: bool = True, fullgraph: bool = False) -> dict[str, Any]:
    """Probe one case: eager reference, ``torch.export`` (unless ``export=False``), then ``torch.compile``."""
    rec: dict[str, Any] = {
        "name": c["name"],
        "group": c["group"],
        "note": c["note"],
        "tags": c["tags"],
        "qualname": qualname(c["target"], c["method"]),
    }
    if c["skip"]:
        rec.update(export="skipped", compile="skipped", skip=c["skip"])
        return rec
    t0 = time.time()
    try:
        wrap, inputs, ref = _reference(c)
    except Exception as e:
        if "PythonDispatcherTLS" in str(e):
            raise Poisoned(rec["name"]) from e
        st = "no-tensor-output" if "NO_TENSOR_OUTPUT" in str(e) else "eager-fail"
        rec.update(export=st, compile=st, error=err_str(e))
        return rec
    if export:
        rec.update(export_step(wrap, inputs))
        rec["export_s"] = round(time.time() - t0, 1)
    rec.update(compile_step(c, wrap, inputs, ref, fullgraph=fullgraph))
    return rec


def main() -> None:
    configure_survey_process()
    if _HAS_ALARM:
        signal.signal(signal.SIGALRM, _alarm)
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = False
    mod = importlib.import_module(sys.argv[1])
    out_json = sys.argv[2]
    only = sys.argv[3:] or None
    cases = mod.CASES
    done = prepare_resume(cases, out_json)
    results: list[dict[str, Any]] = []
    for c in cases:
        if only and not any(o == c["name"] or o == c["group"] or c["name"].startswith(o) for o in only):
            continue
        if c["name"] in done and not only:
            results.append(done[c["name"]])
            continue
        # write a crash marker first so a hard crash (segfault, OOM) is visible on resume
        with open(out_json, "w") as fh:
            json.dump(
                results + [{"name": c["name"], "group": c["group"], "export": "crashed", "compile": "crashed"}],
                fh,
                indent=1,
                default=str,
            )
        try:
            rec = run_case(c)
        except Poisoned:
            # drop the crash marker, keep what is done and let the caller start a fresh interpreter
            kept = {r["name"]: r for r in done.values()} if only else {}
            kept.update({r["name"]: r for r in results})
            with open(out_json, "w") as fh:
                json.dump(list(kept.values()), fh, indent=1, default=str)
            print(f"{c['name']}: interpreter poisoned by an earlier case, exiting for a restart", flush=True)
            sys.exit(POISON_EXIT)
        results.append(rec)
        with open(out_json, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
        print(
            f"{rec['name']:48s} export={rec.get('export'):8s} compile={rec.get('compile'):14s} "
            f"{rec.get('export_s', ''):>6} {rec.get('compile_s', ''):>6} "
            f"{rec.get('export_error') or rec.get('compile_error') or ''}"[:200],
            flush=True,
        )
    if only:
        merged = {r["name"]: r for r in done.values()}
        for r in results:
            merged[r["name"]] = r
        order = [c["name"] for c in cases]
        with open(out_json, "w") as fh:
            json.dump([merged[n] for n in order if n in merged], fh, indent=1, default=str)
    from collections import Counter

    print(
        "SUMMARY export",
        dict(Counter(r.get("export") for r in results)),
        "compile",
        dict(Counter(str(r.get("compile")).split(":")[0] for r in results)),
        flush=True,
    )


if __name__ == "__main__":
    main()
