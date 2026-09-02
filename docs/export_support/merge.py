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

"""Merge the ONNX survey results and the torch.export / torch.compile probe results into
docs/source/_data/export_support.json, the committed snapshot the support page is rendered from.

Usage: python merge.py <onnx_results_glob> <compile_results_glob> <out.json>

Cross-references on the page come from the Sphinx inventory of the last docs build
(docs/build/html/objects.inv) when it exists; without one every operator is rendered as plain text.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_OF = {
    "augmentation": "kornia.augmentation",
    "augmentation3d": "kornia.augmentation",
    "sensors": "kornia.sensors",
}

# aten ops the dynamo ONNX exporter has no lowering for, named the way users call them.
ATEN_NAMES = {
    "_linalg_svd": "torch.linalg.svd",
    "_linalg_solve_ex": "torch.linalg.solve",
    "linalg_lu_factor_ex": "torch.linalg.lu_factor",
    "_linalg_eigh": "torch.linalg.eigh",
    "linalg_qr": "torch.linalg.qr",
    "histc": "torch.histc",
    "searchsorted": "torch.searchsorted",
    "_conj": "complex conj (torch.fft)",
    "reciprocal": "complex reciprocal",
}


def _first_frame(where) -> str:
    if not where:
        return ""
    if isinstance(where, str):
        try:
            where = json.loads(where.replace("'", '"'))
        except Exception:
            return ""
    # the deepest kornia frame is the most useful one
    frame = str(where[-1])
    m = re.match(r"(kornia/[\w/.]+:\d+)", frame)
    return m.group(1) if m else ""


# (substring of the error message, user-facing cause), first match wins. The input shapes are always
# concrete at capture time; the "data-dependent" family is about tensor *values* being read back
# into Python, which no fixed graph can represent.
_CAUSES = [
    (
        "Could not extract specialized integer from data-dependent expression",
        "a Python integer (an index, loop count or size) is computed from tensor values at run time",
    ),
    ("GuardOnDataDependentSymNode", "a Python branch (if / while) tests tensor values at run time"),
    ("Data-dependent branching", "a Python branch (if / while) tests tensor values at run time"),
    ("Dynamic shape operator", "the output shape depends on tensor values (nonzero / boolean indexing)"),
    ("depends on input Tensor data", "the output shape depends on tensor values (nonzero / boolean indexing)"),
    ("torch._C.Generator", "a torch.Generator argument cannot be captured"),
    (
        "Nodes in a graph must be topologically sorted",
        "exporter bug: onnxscript emits an unsorted graph (the eager result is correct)",
    ),
    ("is mutated in the forward method", "a constant buffer is mutated in place during the forward"),
    ("No conversion available yet when dim is None", "no ONNX lowering for the dim=None reduction the operator uses"),
    (
        "Overloaded torch operator invoked from Python failed to match",
        "the exporter cannot resolve the torch overload the operator calls",
    ),
    ("its data is not allocated", "a tensor value is read back into Python (.item() / bool()) during the forward"),
    ("_local_scalar_dense", "a tensor value is read back into Python (.item() / bool()) during the forward"),
]

# torch-version bugs: (all substrings present, cause)
_TORCH_BUGS = [
    (
        ("InductorError", "CalledProcessError"),
        "inductor C++ codegen bug in this torch version (generated kernel does not compile)",
    ),
    (("InductorError", "g++"), "inductor C++ codegen bug in this torch version (generated kernel does not compile)"),
    (
        ("BackendCompilerFailed", "UntypedStorage"),
        "inductor bug in this torch version (KeyError on a tensor storage while lowering the graph)",
    ),
]

_STATUS_CAUSES = {
    "mismatch": "runs, but the result differs from eager: {e}",
    "eager-fail": "does not run eagerly with the probed inputs: {e}",
    "no-tensor-output": "returns no tensor (Python object output), nothing to capture",
    "ort-run-fail": "onnxruntime cannot run the exported graph: {e}",
    "ort-load-fail": "onnxruntime cannot run the exported graph: {e}",
    "checker-fail": "onnx.checker rejects the exported graph: {e}",
    "crashed": "the probe process crashed (segfault / out of memory)",
}


def _reason(status: str, error: str) -> str:
    """Short, user-facing cause of a non-ok status."""
    e = error or ""
    m = re.search(r"No ONNX function found for <OpOverload\(op='aten\.(\w+)'", e)
    if m:
        op = m.group(1)
        return f"no ONNX lowering for {ATEN_NAMES.get(op, 'aten.' + op)}"
    for needle, cause in _CAUSES:
        if needle in e:
            return cause
    if "DataDependentOutputException" in e:
        m = re.search(r"DataDependentOutputException\s*:?\s*(?:<-\s*)?aten\.(\w+)", e)
        return f"{m.group(1) if m else 'an operator'} produces a Python value from tensor values (no graph output)"
    for needles, cause in _TORCH_BUGS:
        if all(n in e for n in needles):
            return cause
    if status in _STATUS_CAUSES:
        return _STATUS_CAUSES[status].format(e=e[:120])
    return e[:160]


def _norm_onnx(r: dict) -> tuple[str, str]:
    s = r["status"]
    err = str(r.get("error", "") or "")
    if s in ("ok", "ok-unverified"):
        return s, "" if s == "ok" else "random operator: exported and ran; only output shape and finiteness are checked"
    if s in ("export-fail", "checker-fail", "ort-run-fail", "ort-load-fail"):
        return "fail", _reason(s, err)
    if s == "mismatch":
        return "mismatch", _reason(s, err)
    return "n/a", _reason(s, err)


def _norm_export(c: dict | None) -> tuple[str, str]:
    if c is None:
        return "n/a", "not probed"
    s = str(c.get("export", ""))
    if s == "ok":
        return "ok", ""
    if s == "fail":
        return "fail", _reason(s, str(c.get("export_error", "")))
    return "n/a", _reason(s, str(c.get("error", c.get("skip", ""))))


def _norm_compile(c: dict | None) -> tuple[str, int, str]:
    if c is None:
        return "n/a", 0, "not probed"
    s = str(c.get("compile", ""))
    breaks = int(c.get("graph_breaks", 0) or 0)
    if s == "ok":
        return "ok", 0, ""
    if s.startswith("ok-breaks"):
        reasons = c.get("break_reasons") or []
        why = ""
        if reasons:
            why = _reason("", str(reasons[0])) or str(reasons[0]).split("\n")[0]
        return "ok-breaks", breaks, f"{breaks} graph break{'s' if breaks != 1 else ''}: {why}".rstrip(": ")
    if s == "ok-unverified":
        return (
            "ok-unverified",
            breaks,
            "random operator: compiled and ran; only output shape and finiteness are checked",
        )
    if s == "mismatch":
        return "mismatch", breaks, _reason(s, str(c.get("compile_error", "")))
    if s == "fail":
        return "fail", breaks, _reason(s, str(c.get("compile_error", "")))
    return "n/a", 0, _reason(s, str(c.get("error", c.get("skip", ""))))


def _split_name(name: str, group: str) -> tuple[str, str]:
    """'transform.warp_perspective[fill]' in group 'geometry.transform' -> ('warp_perspective', 'fill')."""
    parts = set(group.split("."))
    op = name
    changed = True
    while changed:
        changed = False
        for p in parts:
            if op.startswith(p + ".") and len(op) > len(p) + 1:
                op = op[len(p) + 1 :]
                changed = True
    variant = ""
    m = re.match(r"^(.*?)\[(.*)\]$", op)
    if m:
        op, variant = m.group(1), m.group(2).replace("][", ", ")  # 'RandomCrop[p=0.5][random]' -> 'p=0.5, random'
    return op, variant


def _ref_by_name(package: str, op: str, inventory: set[str]) -> str:
    """Shortest documented name under ``package`` that ends in ``op`` (``Boxes.to_mask`` matches a method)."""
    if not re.match(r"^[A-Za-z_][\w.]*$", op):
        return ""
    cands = [n for n in inventory if n.startswith(package + ".") and n.endswith("." + op)]
    return min(cands, key=len) if cands else ""


def _public_ref(qualname: str | None, inventory: set[str]) -> str:
    """Shortest documented alias of an internal qualified name, or '' when the object is not documented."""
    if not qualname:
        return ""
    mod, _, obj = qualname.rpartition(".")
    # methods: module.Class.method -> the object part is Class.method
    parts = mod.split(".")
    for i in range(len(parts)):
        if parts[i].startswith("_"):
            parts = parts[:i]
            break
    obj_parts = qualname.split(".")
    for n_obj in (1, 2):
        obj = ".".join(obj_parts[-n_obj:])
        mod_parts = obj_parts[:-n_obj]
        for k in range(1, len(mod_parts) + 1):
            cand = ".".join(mod_parts[:k] + [obj])
            if cand in inventory:
                return cand
    return ""


REPO = Path(__file__).resolve().parents[2]


def _load_results(pattern: str) -> dict[str, dict]:
    """Records from every file matching ``pattern``, keyed by case name.

    A crash marker, or an eager failure caused by a poisoned interpreter (an earlier case left the
    dispatcher unusable), never overrides a real record from another file.
    """
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f)):
            bogus = "crashed" in (r.get("export"), r.get("status")) or "PythonDispatcherTLS" in str(r.get("error", ""))
            if bogus and r["name"] in out:
                continue
            out[r["name"]] = r
    return out


def _inventory() -> set[str]:
    """Documented object names from the last docs build, used to turn operator names into cross-references."""
    inv = REPO / "docs" / "build" / "html" / "objects.inv"
    if not inv.exists():
        print(f"no {inv}: operators will not be cross-referenced; build the docs first", file=sys.stderr)
        return set()
    from sphinx.util.inventory import InventoryFile

    with open(inv, "rb") as fh:
        data = InventoryFile.load(fh, "", lambda base, loc: loc)
    return {name for objs in data.values() for name in objs}


def main(onnx_glob: str, compile_glob: str, out_path: str) -> None:
    onnx_rs = _load_results(onnx_glob)
    compile_rs = _load_results(compile_glob)
    inventory = _inventory()

    cases = []
    for r in onnx_rs.values():
        c = compile_rs.get(r["name"])
        group = r["group"]
        top = group.split(".")[0]
        package = PACKAGE_OF.get(top, f"kornia.{top}")
        section = group[len(top) + 1 :] if "." in group else ""
        if top == "augmentation3d":
            section = "3d" + ("." + section if section else "")
        section = re.sub(r"\.?random$", "", section)
        op, variant = _split_name(r["name"], group)
        onnx_s, onnx_d = _norm_onnx(r)
        exp_s, exp_d = _norm_export(c)
        comp_s, breaks, comp_d = _norm_compile(c)
        where = _first_frame(r.get("where")) if onnx_s == "fail" else ""
        if not where and c is not None and exp_s == "fail":
            where = _first_frame(c.get("export_where"))
        cases.append(
            {
                "name": r["name"],
                "package": package,
                "section": section,
                "operator": op,
                "variant": variant,
                "ref": _ref_by_name(package, op, inventory) or _public_ref((c or {}).get("qualname"), inventory),
                "note": r.get("note", ""),
                "onnx": onnx_s,
                "onnx_detail": onnx_d,
                "export": exp_s,
                "export_detail": exp_d,
                "compile": comp_s,
                "graph_breaks": breaks,
                "compile_detail": comp_d,
                "where": where,
            }
        )

    import onnx
    import onnxruntime
    import onnxscript
    import torch

    import kornia

    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO, check=False
    ).stdout.strip()
    data = {
        "generated_at": dt.datetime.now(tz=dt.UTC).date().isoformat(),
        "revision": rev,
        "kornia": kornia.__version__,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxscript": onnxscript.__version__,
        "python": platform.python_version(),
        "opset": 18,
        "device": "cpu",
        "cases": cases,
    }
    # the layout pre-commit's pretty-format-json expects, so a regenerated snapshot diffs cleanly
    Path(out_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    from collections import Counter

    print(
        len(cases),
        "onnx",
        dict(Counter(x["onnx"] for x in cases)),
        "export",
        dict(Counter(x["export"] for x in cases)),
        "compile",
        dict(Counter(x["compile"] for x in cases)),
        "refs",
        sum(1 for x in cases if x["ref"]),
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
