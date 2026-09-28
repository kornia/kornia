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
           (torch.compile default inductor backend, fullgraph=False; N = dynamo graph breaks)
plus the target's importable qualified name when it has one.
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
from harness import SEED, _kornia_frames, close, err_str, make_wrapper, prepare_resume, to_np
from torch import nn

torch.set_num_threads(4)
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = False
TIMEOUT = 420


class Timeout(Exception):
    pass


class Poisoned(Exception):
    """A previous case left the dispatcher unusable (PythonDispatcherTLS not set): restart the worker."""


POISON_EXIT = 3


def _alarm(signum, frame):
    raise Timeout(f"case exceeded {TIMEOUT}s")


_HAS_ALARM = hasattr(signal, "SIGALRM")  # POSIX only; on Windows the probes simply run without a timeout
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _alarm)


def _set_alarm(seconds: int) -> None:
    if _HAS_ALARM:
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
        ok, why = close(to_np(a), to_np(b), c["atol"], c["rtol"])
        if not ok:
            bad.append(f"out{i}: {why}")
    return "; ".join(bad)[:400] if bad else None


def _finite(out: list[torch.Tensor], ref: list[torch.Tensor]) -> str | None:
    if len(out) != len(ref):
        return f"{len(out)} outputs vs {len(ref)}"
    for i, (a, b) in enumerate(zip(ref, out)):
        if tuple(a.shape) != tuple(b.shape):
            return f"out{i}: shape {tuple(b.shape)} vs {tuple(a.shape)}"
        if b.is_floating_point() and not torch.isfinite(b).all():
            return f"out{i}: non-finite values"
    return None


def run_case(c: dict[str, Any]) -> dict[str, Any]:
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
        target = c["target"]
        if isinstance(target, nn.Module):
            target.eval()
        wrap = make_wrapper(target, c["kwargs"], len(c["inputs"]), c["method"])
        inputs = [x.detach().clone() for x in c["inputs"]]
        torch.manual_seed(SEED)
        with torch.no_grad():
            ref = list(wrap(*copy.deepcopy(inputs)))
    except Exception as e:
        if "PythonDispatcherTLS" in str(e):
            raise Poisoned(rec["name"]) from e
        st = "no-tensor-output" if "NO_TENSOR_OUTPUT" in str(e) else "eager-fail"
        rec.update(export=st, compile=st, error=err_str(e))
        return rec

    # ---- torch.export
    _set_alarm(TIMEOUT)
    try:
        torch.manual_seed(SEED)
        with torch.no_grad():
            torch.export.export(wrap, tuple(copy.deepcopy(inputs)))
        rec["export"] = "ok"
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        rec.update(export="fail", export_error=err_str(e), export_where=_kornia_frames(e))
    finally:
        _set_alarm(0)
    rec["export_s"] = round(time.time() - t0, 1)

    # ---- torch.compile (dynamo capture stats, then inductor run)
    t1 = time.time()
    _set_alarm(TIMEOUT)
    try:
        torch._dynamo.reset()
        torch.manual_seed(SEED)
        with torch.no_grad():
            ex = torch._dynamo.explain(wrap)(*copy.deepcopy(inputs))
        breaks = int(ex.graph_break_count)
        rec["graph_breaks"] = breaks
        if breaks:
            rec["break_reasons"] = sorted({str(r.reason)[:160] for r in ex.break_reasons})[:5]
        torch._dynamo.reset()
        compiled = torch.compile(wrap)
        torch.manual_seed(SEED)
        with torch.no_grad():
            out = list(compiled(*copy.deepcopy(inputs)))
        if not c["check"]:
            why = _finite(out, ref)
            rec["compile"] = "ok-unverified" if why is None else "mismatch"
        else:
            why = _compare(ref, out, c)
            rec["compile"] = ("ok" if breaks == 0 else f"ok-breaks:{breaks}") if why is None else "mismatch"
        if why:
            rec["compile_error"] = why
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        rec.update(compile="fail", compile_error=err_str(e), compile_where=_kornia_frames(e))
    finally:
        _set_alarm(0)
        torch._dynamo.reset()
    rec["compile_s"] = round(time.time() - t1, 1)
    return rec


def main() -> None:
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
        json.dump(
            results + [{"name": c["name"], "group": c["group"], "export": "crashed", "compile": "crashed"}],
            open(out_json, "w"),
            indent=1,
            default=str,
        )
        try:
            rec = run_case(c)
        except Poisoned:
            # drop the crash marker, keep what is done and let the caller start a fresh interpreter
            kept = {r["name"]: r for r in done.values()} if only else {}
            kept.update({r["name"]: r for r in results})
            json.dump(list(kept.values()), open(out_json, "w"), indent=1, default=str)
            print(f"{c['name']}: interpreter poisoned by an earlier case, exiting for a restart", flush=True)
            sys.exit(POISON_EXIT)
        results.append(rec)
        json.dump(results, open(out_json, "w"), indent=1, default=str)
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
        json.dump([merged[n] for n in order if n in merged], open(out_json, "w"), indent=1, default=str)
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
