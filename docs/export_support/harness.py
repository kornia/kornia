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

"""Shared harness of the library-wide graph-capture survey behind the ONNX / torch.compile / torch.export
support page (docs/source/get-started/export-support.rst, rendered by docs/generate_export_support.py).

A *case* is one public kornia callable (function or nn.Module instance) plus a concrete set of
tensor inputs. Every tensor input becomes an ONNX graph input; Python-only kwargs are baked into
the graph. The harness:

  1. runs the target eagerly (reference),
  2. exports with ``torch.onnx.export(dynamo=True, opset_version=18)``,
  3. runs ``onnx.checker.check_model`` (full check),
  4. runs the graph in onnxruntime (CPU) and compares every tensor output against the reference,
  5. records op types, size, wall time and (on failure) the root-cause error.

Usage from a registry file::

    from harness import case, run_cases
    CASES = [case("rgb_to_grayscale", "color", K.color.rgb_to_grayscale, [torch.rand(1, 3, 32, 32)]), ...]
    if __name__ == "__main__":
        run_cases(CASES, sys.argv[1], only=sys.argv[2:] or None)

Statuses: ok | ok-unverified (check=False, nondeterministic op) | eager-fail | export-fail |
checker-fail | ort-load-fail | ort-run-fail | mismatch | no-tensor-output | skipped.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils._pytree import tree_flatten

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
ort.set_default_logger_severity(3)
torch.set_num_threads(4)

SEED = 1234


def case(
    name: str,
    group: str,
    target: Any,
    inputs: list[torch.Tensor],
    kwargs: dict[str, Any] | None = None,
    *,
    note: str = "",
    check: bool = True,
    atol: float = 2e-4,
    rtol: float = 1e-3,
    skip: str | None = None,
    tags: tuple[str, ...] = (),
    method: str | None = None,
) -> dict[str, Any]:
    """Describe one survey case.

    Args:
        name: unique display name (usually the public callable, e.g. ``"filters.filter3d"``).
        group: report grouping (``"filters"``, ``"geometry.epipolar"``, ``"feature.descriptors"``...).
        target: nn.Module instance or plain callable ``fn(*inputs, **kwargs)``.
        inputs: tensors; each becomes a live ONNX input. Non-tensor positional args go in ``kwargs``
            only when the callable accepts them by keyword; otherwise wrap with a lambda.
        kwargs: Python-only keyword args baked as constants.
        note: free text shown in the report (e.g. "kernel_size baked", "weights: hardnet liberty").
        check: False for ops with randomness inside the graph -> export+run only, output unverified.
        skip: reason string; the case is reported as skipped without running.
        tags: e.g. ("3d", "model", "pretrained").
        method: call this method on the module instead of ``forward`` (e.g. ``"detect"``).
    """
    return {
        "name": name,
        "group": group,
        "target": target,
        "inputs": list(inputs),
        "kwargs": dict(kwargs or {}),
        "note": note,
        "check": check,
        "atol": atol,
        "rtol": rtol,
        "skip": skip,
        "tags": list(tags),
        "method": method,
    }


# ----------------------------------------------------------------------------- wrapping


def _flatten_outputs(out: Any) -> list[torch.Tensor]:
    # dataclass-like kornia containers (Boxes, Keypoints, Se3, Quaternion, Image...) expose tensors via attributes
    if hasattr(out, "data") and isinstance(out.data, torch.Tensor) and not isinstance(out, torch.Tensor):
        return [out.data]
    if hasattr(out, "__dict__") and not isinstance(out, (torch.Tensor, dict, list, tuple)):
        vals = []
        for v in vars(out).values():
            vals += _flatten_outputs(v)
        if vals:
            return vals
    leaves, _ = tree_flatten(out)
    res = []
    for x in leaves:
        if isinstance(x, torch.Tensor):
            res.append(x)
        elif hasattr(x, "data") and isinstance(x.data, torch.Tensor):
            res.append(x.data)
    return res


def make_wrapper(target: Any, kwargs: dict[str, Any], n_inputs: int, method: str | None) -> nn.Module:
    """Return an nn.Module with a fixed-arity forward(in0, ..., inN-1) -> tuple[Tensor, ...]."""
    args = ", ".join(f"in{i}" for i in range(n_inputs))
    src = f"def forward(self, {args}):\n    return self._call(({args},))\n"
    ns: dict[str, Any] = {}
    exec(src, ns)  # noqa: S102

    class Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if isinstance(target, nn.Module):
                self.target = target
            else:
                self._fn = target
            self.kw = kwargs

        def _call(self, xs: tuple) -> tuple:
            t = self.target if isinstance(target, nn.Module) else self._fn
            fn = getattr(t, method) if method else t
            out = fn(*xs, **self.kw)
            outs = _flatten_outputs(out)
            if not outs:
                raise RuntimeError("NO_TENSOR_OUTPUT")
            return tuple(outs)

    Wrap.forward = ns["forward"]
    return Wrap()


# ----------------------------------------------------------------------------- utilities


def to_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.dtype == torch.bfloat16:
        x = x.float()
    return x.numpy()


def close(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> tuple[bool, str]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if bool in (a.dtype, b.dtype):
        ok = bool(np.array_equal(a, b))
        return ok, "" if ok else f"{int((a != b).sum())} bool mismatches"
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    nan_a, nan_b = np.isnan(a64), np.isnan(b64)
    if nan_a.any() or nan_b.any():
        if not np.array_equal(nan_a, nan_b):
            return False, f"NaN pattern differs (torch {int(nan_a.sum())}, ort {int(nan_b.sum())} NaNs)"
        a64, b64 = a64[~nan_a], b64[~nan_b]
    ok = bool(np.allclose(a64, b64, atol=atol, rtol=rtol))
    if ok:
        return True, ""
    diff = np.abs(a64 - b64)
    return (
        False,
        f"max abs diff {diff.max():.3e} at {int((diff > atol + rtol * np.abs(b64)).sum())}/{diff.size} elements",
    )


def op_types(model: onnx.ModelProto) -> list[str]:
    s: set[str] = set()

    def walk(graph: onnx.GraphProto) -> None:
        for n in graph.node:
            s.add(n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}.{n.op_type}")
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                for gg in a.graphs:
                    walk(gg)

    walk(model.graph)
    for f in model.functions:
        for n in f.node:
            s.add(n.op_type if n.domain in ("", "ai.onnx") else f"{n.domain}.{n.op_type}")
    return sorted(x for x in s if not x.startswith("pkg."))


def err_str(e: BaseException) -> str:
    root = e
    while root.__cause__ is not None or root.__context__ is not None:
        root = root.__cause__ or root.__context__  # type: ignore[assignment]
    parts = [f"{type(e).__name__}"]
    msg_src = root if root is not e else e
    msg = [ln for ln in str(msg_src).splitlines() if ln.strip()]
    first = (msg[0] if msg else "")[:240]
    parts.append(f"<- {type(root).__name__}: {first}" if root is not e else f": {first}")
    return " ".join(parts)


def err_full(e: BaseException) -> str:
    out = []
    x: BaseException | None = e
    while x is not None:
        out.append(f"{type(x).__name__}: {str(x)[:2000]}")
        x = x.__cause__ or x.__context__
    return "\n-- caused by --\n".join(out)[-4000:]


def _kornia_frames(e: BaseException) -> list[str]:
    """Locate the kornia source lines involved in a failure (root cause first)."""
    frames: list[str] = []
    x: BaseException | None = e
    while x is not None:
        for fr in traceback.extract_tb(x.__traceback__):
            if "/kornia/kornia/" in fr.filename:
                short = fr.filename.split("/kornia/kornia/")[-1]
                frames.append(f"kornia/{short}:{fr.lineno} {fr.line.strip() if fr.line else ''}")
        x = x.__cause__ or x.__context__
    # dedupe, keep order, innermost last
    seen: set[str] = set()
    res = [f for f in frames if not (f in seen or seen.add(f))]
    return res[-6:]


def export(m: nn.Module, inputs: list[torch.Tensor]) -> onnx.ModelProto:
    import contextlib

    buf = io.BytesIO()
    # dynamo prints the whole FX graph on data-dependent failures; keep the survey log readable
    with torch.no_grad(), open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        prog = torch.onnx.export(
            m,
            tuple(inputs),
            dynamo=True,
            opset_version=18,
            verbose=False,
            report=False,
            optimize=os.environ.get("KORNIA_SURVEY_OPTIMIZE", "1") == "1",
        )
    prog.save(buf)
    return onnx.load_from_string(buf.getvalue())


# ----------------------------------------------------------------------------- runner


def run_case(c: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {k: c[k] for k in ("name", "group", "note", "tags", "check")}
    rec["n_inputs"] = len(c["inputs"])
    rec["input_shapes"] = [list(x.shape) + [str(x.dtype).replace("torch.", "")] for x in c["inputs"]]
    rec["baked"] = {k: repr(v)[:60] for k, v in c["kwargs"].items()}
    if c["skip"]:
        rec.update(status="skipped", error=c["skip"])
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
            ref = wrap(*copy.deepcopy(inputs))
        rec["n_outputs"] = len(ref)
        rec["output_shapes"] = [list(x.shape) + [str(x.dtype).replace("torch.", "")] for x in ref]
    except Exception as e:
        status = "no-tensor-output" if "NO_TENSOR_OUTPUT" in str(e) else "eager-fail"
        rec.update(status=status, error=err_str(e), error_full=err_full(e), where=_kornia_frames(e))
        return rec
    try:
        torch.manual_seed(SEED)
        model = export(wrap, inputs)
    except Exception as e:
        rec.update(
            status="export-fail",
            error=err_str(e),
            error_full=err_full(e),
            where=_kornia_frames(e),
            time_s=round(time.time() - t0, 1),
        )
        return rec
    blob = model.SerializeToString()
    rec["size_kb"] = len(blob) // 1024
    rec["ops"] = op_types(model)
    rec["time_s"] = round(time.time() - t0, 1)
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        rec.update(status="checker-fail", error=err_str(e), error_full=err_full(e))
        return rec
    try:
        sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    except Exception as e:
        rec.update(status="ort-load-fail", error=err_str(e), error_full=err_full(e))
        return rec
    names = [i.name for i in sess.get_inputs()]
    rec["graph_inputs"] = names
    feed = {}
    dropped = []
    for i, x in enumerate(inputs):
        n = f"in{i}"
        if n in names:
            feed[n] = to_np(x)
        else:
            dropped.append(n)
    if len(feed) != len(names):
        # exporter renamed inputs: fall back to positional feed when the count matches
        if len(names) == len(inputs):
            feed = {n: to_np(x) for n, x in zip(names, inputs)}
            dropped = []
        else:
            rec.update(
                status="ort-run-fail", error=f"cannot map inputs: graph has {names}, case has {len(inputs)} tensors"
            )
            return rec
    rec["dropped_inputs"] = dropped
    try:
        outs = sess.run(None, feed)
    except Exception as e:
        rec.update(status="ort-run-fail", error=err_str(e), error_full=err_full(e))
        return rec
    if not c["check"]:
        rec["status"] = "ok-unverified"
        return rec
    if len(outs) != len(ref):
        rec.update(status="mismatch", error=f"{len(outs)} ORT outputs vs {len(ref)} torch outputs")
        return rec
    bad = []
    for i, (a, b) in enumerate(zip(ref, outs)):
        ok, why = close(to_np(a), b, c["atol"], c["rtol"])
        if not ok:
            bad.append(f"out{i}: {why}")
    if bad:
        rec.update(status="mismatch", error="; ".join(bad)[:400])
        return rec
    rec["status"] = "ok"
    return rec


def run_cases(cases: list[dict[str, Any]], out_json: str, only: list[str] | None = None, resume: bool = True) -> None:
    names = [c["name"] for c in cases]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate case names: {sorted(dup)}")
    done: dict[str, dict[str, Any]] = {}
    if resume and os.path.exists(out_json):
        try:
            done = {r["name"]: r for r in json.load(open(out_json))}
        except Exception:
            done = {}
    results: list[dict[str, Any]] = []
    for c in cases:
        if only and not any(o == c["name"] or o == c["group"] or c["name"].startswith(o) for o in only):
            continue
        if c["name"] in done and done[c["name"]].get("status") not in (None, "crashed") and not only:
            results.append(done[c["name"]])
            continue
        # mark as crashed first so a hard crash (segfault) is visible after restart
        json.dump(
            results + [{"name": c["name"], "group": c["group"], "status": "crashed", "note": c["note"]}],
            open(out_json, "w"),
            indent=1,
            default=str,
        )
        rec = run_case(c)
        results.append(rec)
        json.dump(results, open(out_json, "w"), indent=1, default=str)
        extra = rec.get("error", "")
        if rec["status"] == "ok" and rec.get("dropped_inputs"):
            extra = f"inputs folded away: {rec['dropped_inputs']}"
        print(
            f"{rec['name']:48s} {rec['status']:14s} {rec.get('size_kb', ''):>6} {rec.get('time_s', ''):>6} {extra}",
            flush=True,
        )
    if only:
        # merge into existing file without disturbing other records
        merged = {r["name"]: r for r in done.values()}
        for r in results:
            merged[r["name"]] = r
        order = [c["name"] for c in cases]
        json.dump(
            [merged[n] for n in order if n in merged] + [r for n, r in merged.items() if n not in order],
            open(out_json, "w"),
            indent=1,
            default=str,
        )
    from collections import Counter

    print("SUMMARY", dict(Counter(r["status"] for r in results)), flush=True)
