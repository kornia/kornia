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

"""Render docs/source/get-started/export-support.rst from docs/source/_data/export_support.json.

The JSON is a committed snapshot of the graph-capture survey in ``docs/export_support/``: every public
operator, model and augmentation probed with the dynamo ONNX exporter, ``torch.export`` and
``torch.compile``. Refresh it with ``python docs/export_support/run.py`` (see that package's README)
and commit the result; this module turns it into the support page at docs-build time, the same way
``generate_adoption.py`` renders the Adoption page.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "source" / "_data" / "export_support.json"
OUT = REPO / "docs" / "source" / "get-started" / "export-support.rst"

# Display order of the major packages; anything else follows alphabetically.
PACKAGE_ORDER = [
    "kornia.color",
    "kornia.enhance",
    "kornia.filters",
    "kornia.morphology",
    "kornia.geometry",
    "kornia.feature",
    "kornia.augmentation",
    "kornia.losses",
    "kornia.metrics",
    "kornia.contrib",
    "kornia.models",
    "kornia.image",
    "kornia.sensors",
    "kornia.utils",
    "kornia.tracking",
]

# Section headings inside a package, keyed by the survey's group suffix.
SECTION_TITLES = {
    ("kornia.augmentation", ""): "2D augmentations",
    ("kornia.augmentation", "3d"): "3D augmentations",
    ("kornia.augmentation", "container"): "Containers",
    ("kornia.augmentation", "mix"): "Mix augmentations",
    ("kornia.augmentation", "auto"): "Auto-augment policies",
    ("kornia.contrib", ""): "Operators",
    ("kornia.contrib", "patches"): "Tensor patches",
    ("kornia.contrib", "wrappers"): "Detector and model wrappers",
    ("kornia.feature", "detectors"): "Detectors",
    ("kornia.feature", "descriptors"): "Descriptors",
    ("kornia.feature", "laf"): "Local affine frames",
    ("kornia.feature", "matching"): "Matching",
    ("kornia.feature", "models"): "Learned detectors and matchers",
    ("kornia.feature", "orientation_affine"): "Orientation and affine shape",
    ("kornia.feature", "responses"): "Corner and blob responses",
    ("kornia.geometry", "bbox"): "Bounding boxes (bbox)",
    ("kornia.geometry", "boxes"): "Boxes",
    ("kornia.geometry", "calibration"): "Calibration",
    ("kornia.geometry", "camera"): "Camera",
    ("kornia.geometry", "conversions"): "Conversions",
    ("kornia.geometry", "depth"): "Depth",
    ("kornia.geometry", "epipolar"): "Epipolar geometry",
    ("kornia.geometry", "grid"): "Grid",
    ("kornia.geometry", "homography"): "Homography",
    ("kornia.geometry", "keypoints"): "Keypoints",
    ("kornia.geometry", "liegroup"): "Lie groups",
    ("kornia.geometry", "line"): "Lines",
    ("kornia.geometry", "linalg"): "Linear algebra",
    ("kornia.geometry", "plane"): "Planes",
    ("kornia.geometry", "pointcloud"): "Point clouds",
    ("kornia.geometry", "pose"): "Pose",
    ("kornia.geometry", "quaternion"): "Quaternion",
    ("kornia.geometry", "ransac"): "RANSAC",
    ("kornia.geometry", "ray"): "Rays",
    ("kornia.geometry", "solvers"): "Polynomial solvers",
    ("kornia.geometry", "subpix"): "Sub-pixel",
    ("kornia.geometry", "transform"): "Transforms",
    ("kornia.geometry", "vector"): "Vectors",
    ("kornia.models", "base"): "Base classes",
    ("kornia.models", "processors"): "Pre- and post-processors",
    ("kornia.sensors", "camera"): "Camera models",
}

# Status -> (docutils role, cell text, meaning shown in the legend).
STATUS = {
    "ok": ("compat-ok", "yes", "captured, and the result matches eager execution"),
    "ok-breaks": (
        "compat-ok-breaks",
        "yes, graph breaks",
        "compiles and matches eager, but dynamo falls back to Python at one or more graph breaks",
    ),
    "ok-unverified": (
        "compat-ok-random",
        "yes (random)",
        "captured and ran; a random operator, so only output shape and finiteness are compared",
    ),
    "mismatch": ("compat-mismatch", "mismatch", "captured and ran, but the result differs from eager execution"),
    "fail": ("compat-fail", "no", "capture or conversion failed; the Details column says why"),
    "n/a": (
        "compat-na",
        "n/a",
        "not a tensor graph (file I/O, enums, Python containers, stubs) or not runnable with the probed inputs",
    ),
}

SUPPORTED = {"ok", "ok-breaks", "ok-unverified"}


def _applicable(case: dict) -> bool:
    """False for the registry entries that are not tensor graphs at all (file I/O, enums, Python containers)."""
    return any(case[key] != "n/a" for key in ("onnx", "export", "compile"))


def load_data() -> dict:
    data = json.loads(DATA.read_text())
    data["cases"] = [c for c in data["cases"] if _applicable(c)]
    return data


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")


def _cell(status: str, breaks: int = 0) -> str:
    role, text, _ = STATUS[status]
    if status == "ok-breaks" and breaks:
        text = f"yes, {breaks} break{'s' if breaks != 1 else ''}"
    return f":{role}:`{text}`"


def _escape(text: str) -> str:
    # Table cells are one line of inline RST: neutralise the markup characters an error message may carry.
    text = " ".join(text.split())
    return text.replace("\\", "\\\\").replace("`", "'").replace("*", "\\*").replace("|", "\\|").replace("_", "\\_")


def _operator_cell(case: dict) -> str:
    op = case["operator"] or case["name"]
    ref = case.get("ref")
    if ref:
        # ``~`` shows only the last component; the full path is in the link target.
        label = f":py:obj:`{op} <{ref}>`"
    else:
        label = f"``{op}``"
    if case.get("variant") and case["operator"]:
        label += f" ``[{case['variant']}]``"
    return label


def _details(case: dict) -> str:
    # columns that share one cause are named together: "ONNX, export, compile: ..."
    by_detail: dict[str, list[str]] = {}
    for column, key in (("ONNX", "onnx"), ("export", "export"), ("compile", "compile")):
        detail = case.get(f"{key}_detail", "")
        if detail:
            by_detail.setdefault(detail, []).append(column)
    parts = [f"**{', '.join(columns)}:** {_escape(detail)}" for detail, columns in by_detail.items()]
    if case.get("where"):
        parts.append(f"``{case['where']}``")
    return " · ".join(parts)


def _table(cases: list[dict]) -> str:
    rows = [
        "   * - Operator",
        "     - ONNX",
        # U+200B lets the header break after "torch." when the column is narrow
        "     - torch.\u200bexport",
        "     - torch.\u200bcompile",
        "     - Details",
    ]
    for c in cases:
        rows += [
            f"   * - {_operator_cell(c)}",
            f"     - {_cell(c['onnx'])}",
            f"     - {_cell(c['export'])}",
            f"     - {_cell(c['compile'], c.get('graph_breaks', 0))}",
            f"     - {_details(c) or ' '}",
        ]
    return "\n".join(rows)


def _counts(cases: list[dict]) -> dict[str, tuple[int, int]]:
    """Per column: (supported, applicable) where applicable excludes n/a rows."""
    out = {}
    for key in ("onnx", "export", "compile"):
        applicable = [c for c in cases if c[key] != "n/a"]
        out[key] = (sum(1 for c in applicable if c[key] in SUPPORTED), len(applicable))
    return out


def _operator_counts(cases: list[dict]) -> dict[str, int]:
    """Operators rather than configurations: how many distinct callables, and how many export to ONNX.

    An operator counts as exporting when at least one of its probed configurations does, so an
    augmentation whose ``[random]`` variant fails but whose deterministic variant passes is counted
    as supported -- that variant is what a user who wants an ONNX file would export.
    """
    ops: dict[tuple[str, str], bool] = {}
    for c in cases:
        key = (c["package"], c["operator"])
        ops[key] = ops.get(key, False) or c["onnx"] in SUPPORTED
    return {"operators": len(ops), "onnx_operators": sum(ops.values())}


def _ratio(pair: tuple[int, int]) -> str:
    good, total = pair
    return f"{good} / {total}" if total else "—"


def _sort_key(case: dict) -> tuple:
    return (case["operator"].lower(), case.get("variant", ""))


def _package_sections(cases: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """[(section key, title, cases)] for one package, tables in a stable order."""
    by_section: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_section[c.get("section", "")].append(c)
    package = cases[0]["package"]
    out = []
    for key in sorted(by_section, key=lambda k: (k != "", k)):
        title = SECTION_TITLES.get((package, key), key.replace("_", " ").capitalize() or "Operators")
        out.append((key, title, sorted(by_section[key], key=_sort_key)))
    return out


def _summary_table(packages: list[str], by_package: dict[str, list[dict]]) -> str:
    rows = [
        "   * - Package",
        "     - Probed",
        "     - ONNX",
        "     - torch.export",
        "     - torch.compile",
    ]
    for pkg in packages:
        cs = by_package[pkg]
        n = _counts(cs)
        rows += [
            f"   * - :ref:`{pkg} <export-support-{_slug(pkg)}>`",
            f"     - {len(cs)}",
            f"     - {_ratio(n['onnx'])}",
            f"     - {_ratio(n['export'])}",
            f"     - {_ratio(n['compile'])}",
        ]
    return "\n".join(rows)


def _legend(cases: list[dict]) -> str:
    used = {c[key] for c in cases for key in ("onnx", "export", "compile")}
    rows = ["   * - Cell", "     - Meaning"]
    for status, (role, text, meaning) in STATUS.items():
        if status in used:
            rows += [f"   * - :{role}:`{text}`", f"     - {meaning}"]
    return "\n".join(rows)


def render_page(data: dict) -> str:
    cases = data["cases"]
    by_package: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_package[c["package"]].append(c)
    packages = [p for p in PACKAGE_ORDER if p in by_package] + sorted(p for p in by_package if p not in PACKAGE_ORDER)
    total = _counts(cases)
    roles = "\n".join(f".. role:: {role}" for role, _, _ in STATUS.values())
    n_ok_all = sum(
        1 for c in cases if c["onnx"] in SUPPORTED and c["export"] in SUPPORTED and c["compile"] in SUPPORTED
    )
    ops = _operator_counts(cases)
    description = (
        f"Which Kornia operators export to ONNX, capture with torch.export and run under torch.compile: "
        f"{len(cases)} probed configurations across {len(packages)} packages, with the reason for every failure."
    )

    sections = []
    for pkg in packages:
        cs = by_package[pkg]
        n = _counts(cs)
        n_fail = Counter()
        for c in cs:
            for key in ("onnx", "export", "compile"):
                if c[key] in ("fail", "mismatch"):
                    n_fail[key] += 1
        lead = (
            f"{len(cs)} probed configurations. ONNX {_ratio(n['onnx'])}, torch.export {_ratio(n['export'])}, "
            f"torch.compile {_ratio(n['compile'])} (supported / probed)."
        )
        block = [f".. _export-support-{_slug(pkg)}:", "", pkg, "-" * len(pkg), "", lead, ""]
        parts = _package_sections(cs)
        for _key, title, section_cases in parts:
            if len(parts) > 1:
                block += [title, "^" * len(title), ""]
            block += [
                ".. list-table::",
                "   :header-rows: 1",
                "   :widths: auto",
                "   :class: kornia-compat-table",
                "",
                _table(section_cases),
                "",
            ]
        sections.append("\n".join(block))

    return f"""\
.. This page is generated at build time by docs/generate_export_support.py from
   docs/source/_data/export_support.json. Edit those, not this file.

{roles}

.. _export-support:

ONNX, torch.compile and torch.export support
============================================

.. meta::
   :description: {description}

Every public operator, model and augmentation in kornia was run through the three graph-capture
paths PyTorch offers, and this page lists the result for each one, package by package, with the
cause of every failure. Use the search box to find an operator, or the summary table to jump to a
package.

.. grid:: 1 3 3 3
   :gutter: 3
   :class-container: kornia-cards kornia-compat-cards

   .. grid-item-card:: ONNX export
      :class-card: kornia-compat-card

      .. rst-class:: kornia-compat-count

      {_ratio(total["onnx"])}

      ``torch.onnx.export(dynamo=True)`` at opset {data.get("opset", 18)}, checked with ``onnx.checker`` and run in
      onnxruntime against eager

   .. grid-item-card:: torch.export
      :class-card: kornia-compat-card

      .. rst-class:: kornia-compat-count

      {_ratio(total["export"])}

      ``torch.export.export`` captures the whole program as one graph

   .. grid-item-card:: torch.compile
      :class-card: kornia-compat-card

      .. rst-class:: kornia-compat-count

      {_ratio(total["compile"])}

      ``torch.compile`` with the default inductor backend, compared with eager; graph breaks counted

Each figure is *supported / probed*; {n_ok_all} of the {len(cases)} probed configurations pass
all three paths; the configurations cover {ops["operators"]} distinct operators, of which
{ops["onnx_operators"]} export to ONNX in at least one configuration. Every operator is captured
with concrete input shapes, so a failure is never about an unknown input size: the causes in the
Details column are missing lowerings, or Python code that reads tensor *values* (an index, a loop
count, an ``if``) while the graph is being traced.

**How this was measured.** Snapshot taken on {data.get("generated_at", "")} with kornia
{data.get("kornia", "")}, torch {data.get("torch", "")}, onnx {data.get("onnx", "")}, onnxruntime
{data.get("onnxruntime", "")} and onnxscript {data.get("onnxscript", "")} on CPU (Python
{data.get("python", "")}). A *configuration* is one public callable plus one concrete set of inputs
and keyword arguments; operators with several code paths (``padding_mode``, ``align_corners``, a
batch of one versus several, random versus fixed parameters) appear once per path, the path named
in brackets after the operator. Augmentations are probed twice: with their sampled parameters fed
in as graph inputs (a deterministic graph) and with the sampling inside the graph (``[random]``).
Registry entries that are not tensor graphs at all (file I/O, enums, Python containers, stubs) are
not listed.
The survey lives in ``docs/export_support/`` and this page is rendered from its committed result,
so the figures follow the library rather than a hand-maintained list.

Legend
------

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :class: kornia-compat-legend

{_legend(cases)}

A failure on this page is a fact about a specific torch, exporter and kornia version, not a
promise: most *no* entries are missing ONNX lowerings upstream (``torch.linalg.svd``, ``solve``,
``lu_factor``, ``eigh``, ``qr``, ``histc``), data-dependent output sizes (matchers that return a
variable number of pairs, detectors that keep the top-k responses above a threshold, NMS), or
Python control flow that reads tensor values. See :doc:`onnx` for how to chain exported graphs and
:doc:`gpu-acceleration` for ``torch.compile`` in practice.

.. raw:: html

   <div class="kornia-compat-controls" hidden>
     <label class="kornia-compat-search">
       <span class="visually-hidden">Find an operator</span>
       <input type="search" placeholder="Find an operator, e.g. warp_perspective" autocomplete="off">
     </label>
     <label class="kornia-compat-filter">
       Show
       <select>
         <option value="all">every row</option>
         <option value="any-fail">rows with a failure or mismatch</option>
         <option value="onnx-fail">ONNX failures</option>
         <option value="export-fail">torch.export failures</option>
         <option value="compile-fail">torch.compile failures</option>
         <option value="compile-breaks">torch.compile graph breaks</option>
         <option value="all-ok">rows that pass all three</option>
       </select>
     </label>
     <span class="kornia-compat-status" aria-live="polite"></span>
   </div>

Summary by package
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15
   :class: kornia-compat-summary

{_summary_table(packages, by_package)}

{chr(10).join(sections)}
"""


def main() -> dict[str, int | str]:
    data = load_data()
    OUT.write_text(render_page(data))
    cases = data["cases"]
    n = _counts(cases)
    return {
        "cases": len(cases),
        "onnx": n["onnx"][0],
        "export": n["export"][0],
        "compile": n["compile"][0],
        "generated_at": str(data.get("generated_at", "")),
        **_operator_counts(cases),
    }


if __name__ == "__main__":
    print(main())
