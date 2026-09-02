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

"""Regenerate the snapshot behind the ONNX / torch.compile / torch.export support page.

Usage (from the repository root, with the docs built once so cross-references resolve):

    python docs/export_support/run.py              # everything: ~2 h on 8 cores, CPU only
    python docs/export_support/run.py aug misc     # only these case groups
    python docs/export_support/run.py --merge-only # re-merge existing results into the JSON

Each case group ``cases_<group>.py`` is probed twice, in parallel subprocesses: once for ONNX
(``harness.run_cases``) and once for torch.export + torch.compile (``probe_compile.py``). Results
land in ``results/`` next to this file (git-ignored) and are resumable: a rerun skips the cases that
already have a record, so a crashed worker can simply be started again. The final merge writes
``docs/source/_data/export_support.json``, which is committed and rendered by
``docs/generate_export_support.py`` at docs build time.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
SNAPSHOT = REPO / "docs" / "source" / "_data" / "export_support.json"
GROUPS = ["aug", "feature", "geomA", "geomB", "misc", "models"]
MAX_RESTARTS = 50
POISON_EXIT = 3  # probe_compile.POISON_EXIT: a case left the interpreter unusable, resume in a fresh one


def _jobs(groups: list[str]) -> list[tuple[str, list[str]]]:
    jobs = []
    for g in groups:
        jobs.append((f"onnx:{g}", [sys.executable, f"cases_{g}.py", str(RESULTS / f"onnx_{g}.json")]))
        jobs.append(
            (f"compile:{g}", [sys.executable, "probe_compile.py", f"cases_{g}", str(RESULTS / f"compile_{g}.json")])
        )
    return jobs


def _worker(label: str, cmd: list[str]) -> tuple[str, int]:
    """Run one probe process to completion, restarting it (it resumes) when a case poisoned the interpreter."""
    log_path = RESULTS / f"{label.replace(':', '_')}.log"
    print(f"started {label}", flush=True)
    with open(log_path, "a") as log:
        for _restart in range(MAX_RESTARTS):
            rc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            if rc != POISON_EXIT:
                break
            print(f"restarting {label}: a case left the interpreter unusable", flush=True)
    print(f"finished {label} (exit {rc})", flush=True)
    return label, rc


def main(argv: list[str]) -> int:
    merge_only = "--merge-only" in argv
    groups = [a for a in argv if not a.startswith("--")] or GROUPS
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        sys.exit(f"unknown case group(s) {unknown}; choose from {GROUPS}")
    RESULTS.mkdir(exist_ok=True)

    if not merge_only:
        with ThreadPoolExecutor(max_workers=len(groups) * 2) as pool:
            failed = [label for label, rc in pool.map(lambda job: _worker(*job), _jobs(groups)) if rc != 0]
        if failed:
            print(
                f"workers exited abnormally: {failed}; rerun to resume the cases they did not record", file=sys.stderr
            )

    cmd = [sys.executable, "merge.py", str(RESULTS / "onnx_*.json"), str(RESULTS / "compile_*.json"), str(SNAPSHOT)]
    return subprocess.run(cmd, cwd=HERE, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
