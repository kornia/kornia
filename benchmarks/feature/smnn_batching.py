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
"""Public SMNN APIs: independent pair calls versus batched descriptor matching.

Inputs stay on the measured device; no extraction, transfers, HDF5, or model
loading is included. Both APIs use identical descriptor pairs, thresholds,
float32 precision, and threads. Synthetic descriptors have 25% noisy positives.
Run this same harness in both the base checkout and the changed worktree;
the base records the unavailable additive API rather than a replacement.
"""

from __future__ import annotations

# Imports follow path setup so direct execution always measures this checkout.
# ruff: noqa: E402
import argparse
import hashlib
import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import torch
from common import kornia_provenance, run_metadata, save_json, setup_run, time_us

import kornia
from kornia import feature


def independent_matches(a: torch.Tensor, b: torch.Tensor):
    return [feature.match_smnn(x, y, 0.95) for x, y in zip(a, b)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--counts", default="512,2048,4096")
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device, dtype, sync = setup_run(args, opencv=False)
    torch.set_float32_matmul_precision("highest")
    source, module = kornia_provenance(kornia.__file__)
    print(f"Kornia source: {source}\nInterpreter: {sys.executable}", flush=True)
    if module == "outside-checkout":
        raise RuntimeError("Kornia imported from a different checkout")
    metadata = {
        **run_metadata(device),
        "kornia_module": module,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matching_sha256": hashlib.sha256((REPO / "kornia/feature/matching.py").read_bytes()).hexdigest(),
        "units": "pairs/s",
        "regime": __doc__,
    }
    rows = []
    with torch.inference_mode():
        for count in [int(n) for n in args.counts.split(",")]:
            for batch in [int(n) for n in args.batches.split(",")]:
                a = torch.nn.functional.normalize(torch.randn(batch, count, 64, device=device, dtype=dtype), dim=-1)
                b = torch.randn_like(a)
                b[:, : count // 4] = a[:, : count // 4] + 0.01 * torch.randn_like(a[:, : count // 4])
                b = torch.nn.functional.normalize(b, dim=-1)

                independent = partial(independent_matches, a, b)
                expected = independent()
                batched = getattr(feature, "match_smnn_batched", None)
                for label, fn in [
                    ("independent", independent),
                    ("batched", partial(batched, a, b, 0.95) if batched else None),
                ]:
                    if label == "batched" and batched is None:
                        rows.append(
                            {
                                "backend": label,
                                "batch": batch,
                                "count": count,
                                "median_us": None,
                                "error": "unavailable",
                            }
                        )
                        continue
                    if label == "batched":
                        ratios, indices = fn()
                        for i, (reference_ratio, reference_indices) in enumerate(expected):
                            selected = indices[:, 0] == i
                            torch.testing.assert_close(indices[selected, 1:], reference_indices)
                            torch.testing.assert_close(ratios[selected], reference_ratio, rtol=2e-5, atol=2e-5)
                    median, iqr = time_us(fn, min_run_time=args.min_run_time, sync=sync)
                    peak = None
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        fn()
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                    rows.append(
                        {
                            "backend": label,
                            "batch": batch,
                            "count": count,
                            "dtype": args.dtype,
                            "median_us": median,
                            "iqr_us": iqr,
                            "throughput_per_s": batch * 1e6 / median,
                            "peak_allocated_bytes": peak,
                            "equivalent_matches": True,
                        }
                    )
                    print(
                        f"N={count} B={batch} {label}: {median / batch:.1f} us/pair (IQR batch {iqr:.1f})", flush=True
                    )
                    save_json(args.json, metadata, rows)
    save_json(args.json, metadata, rows)


if __name__ == "__main__":
    main()
