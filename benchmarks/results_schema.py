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

"""Schema validation for committed benchmark result JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED_METADATA = ("timestamp_utc", "git_commit", "platform", "python", "torch", "kornia", "device", "load")

#: An absolute path in a committed file discloses a home directory or machine layout. The README
#: promises contributors that only aggregate numbers are recorded, so enforce it for every string
#: metadata value rather than only for the keys that exist today.
_ABSOLUTE_PATH = re.compile(r"^(/|\\\\|[A-Za-z]:[\\/])")


def _load_payload(path: Path) -> tuple[dict | None, list[str]]:
    """Parse a result file down to its ``metadata``/``results`` envelope, or explain why it cannot be."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"unreadable JSON: {exc}"]
    if not isinstance(payload, dict) or set(payload) != {"metadata", "results"}:
        return None, [f"top level must be an object with keys metadata/results, got {type(payload).__name__}"]
    if not isinstance(payload["metadata"], dict):
        return None, [f"metadata must be an object, got {type(payload['metadata']).__name__}"]
    return payload, []


def _content_errors(payload: dict, *, require_load: bool) -> list[str]:
    """Envelope, metadata, privacy and row-type rules shared by release snapshots and artefacts."""
    errors: list[str] = []
    meta, results = payload["metadata"], payload["results"]
    for key in REQUIRED_METADATA:
        if key not in meta and (require_load or key != "load"):
            errors.append(f"metadata missing '{key}'")
    load = meta.get("load", {})
    if not isinstance(load, dict) or not all(v is None or isinstance(v, (int, float)) for v in load.values()):
        errors.append("metadata.load must contain only numeric aggregates or null (privacy rule)")
    for key, value in meta.items():
        if isinstance(value, str) and _ABSOLUTE_PATH.match(value):
            errors.append(f"metadata.{key} must not be an absolute filesystem path (privacy rule)")
    if not isinstance(results, list) or not results:
        errors.append("results must be a non-empty list")
    else:
        for i, row in enumerate(results):
            if not isinstance(row, dict):
                errors.append(f"results[{i}] must be an object, got {type(row).__name__}")
                continue
            for key, types in (("op", str), ("backend", str), ("batch", int)):
                if not isinstance(row.get(key), types) or isinstance(row.get(key), bool):
                    errors.append(f"results[{i}].{key} missing or wrong type")
            for key in ("median_us", "throughput_per_s"):
                if (
                    key not in row
                    or isinstance(row[key], bool)
                    or (row[key] is not None and not isinstance(row[key], (int, float)))
                ):
                    errors.append(f"results[{i}].{key} missing or wrong type")
    return errors


def validate_artefact(path: Path) -> list[str]:
    """Return a list of problems with a committed comparison artefact; empty list means valid.

    A comparison artefact is the raw JSON of a base-versus-branch A/B or a cross-release run, kept
    next to the report that cites it (``benchmarks/<area>/<name>_results/``). It follows the content
    rules of a release snapshot -- the envelope, the recorded metadata, the privacy rule and the row
    types -- but not the release layout: its filename names the compared revision rather than a
    machine, and it does not sit in a ``<kornia-version>`` directory, because the revision it records
    is by construction not the current tree. ``load`` is optional because a base revision's harness
    may predate that field.
    """
    payload, errors = _load_payload(path)
    if payload is None:
        return errors
    return _content_errors(payload, require_load=False)


def validate_result(path: Path) -> list[str]:
    """Return a list of problems with a committed result file; empty list means valid."""
    payload, errors = _load_payload(path)
    if payload is None:
        return errors
    errors = _content_errors(payload, require_load=True)
    meta = payload["metadata"]
    name = Path(path).name
    parts = name[: -len(".json")].split("--") if name.endswith(".json") else []
    if len(parts) != 3:
        errors.append("filename must be <suite>--<machine-slug>--<device>.json")
    else:
        device_type = str(meta.get("device", "")).split(":")[0]
        if parts[2] != device_type:
            errors.append(f"filename device '{parts[2]}' != metadata device '{device_type}'")
    if "kornia" in meta and Path(path).parent.name != str(meta["kornia"]):
        errors.append(f"version dir '{Path(path).parent.name}' != metadata kornia '{meta['kornia']}'")
    return errors
