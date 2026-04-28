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

# Licensed under the Apache License, Version 2.0 (the "License").
"""Generate numerical golden snapshots for kornia.augmentation transforms.

Run manually (from the repo root):
    .venv/bin/python -m tests.augmentation.goldens._generate

Output: tests/augmentation/goldens/data/<transform>__seed<N>__<shape>.npz
Each .npz contains:
  - input: the seeded input tensor
  - output: the transform's output
  - seed: the seed used
  - shape: the input shape
  - transform_class: class name
  - kornia_version: package version at generation time
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

import kornia
import kornia.augmentation as K

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def discover() -> list[tuple[str, type]]:
    """Return concrete public transforms with no required constructor args."""
    out = []
    for name in sorted(dir(K)):
        if name.startswith("_"):
            continue
        obj = getattr(K, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, torch.nn.Module):
            continue
        # Skip abstract bases, containers, generators
        if any(
            name.endswith(s)
            for s in (
                "Sequential",
                "Base",
                "Base2D",
                "Base3D",
                "Dispather",  # typo in kornia public name
                "Dispatcher",
            )
        ):
            continue
        if "Generator" in name:
            continue
        try:
            sig = inspect.signature(obj.__init__)
            req = [
                p
                for p in sig.parameters.values()
                if p.name != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if req:
                continue
        except (ValueError, TypeError):
            continue
        out.append((name, obj))
    return out


SEEDS = [0, 42]
SHAPES = [(2, 3, 32, 32), (1, 3, 64, 64)]  # small to keep .npz files tiny


def generate_one(
    name: str,
    cls: type,
    seed: int,
    shape: tuple[int, ...],
) -> tuple[str, str]:
    out_path = DATA_DIR / f"{name}__seed{seed}__{'x'.join(map(str, shape))}.npz"
    try:
        torch.manual_seed(seed)
        x = torch.rand(*shape)
        # Save the input BEFORE running the transform -- some transforms use in-place
        # ops (add_) that would mutate x in-place and corrupt the saved snapshot.
        x_np = x.numpy().copy()
        # Capture RNG state after x is generated so the test can restore it
        # exactly and the transform forward sees the same RNG draws.
        rng_state_after_x: bytes = torch.get_rng_state().numpy().tobytes()
        transform = cls()
        with torch.no_grad():
            y = transform(x)
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy().copy()
        else:
            # Some transforms return tuple/list
            y_np = np.asarray(
                [yi.cpu().numpy().copy() if isinstance(yi, torch.Tensor) else yi for yi in y],
                dtype=object,
            )
        np.savez_compressed(
            out_path,
            input=x_np,
            output=y_np,
            seed=np.int64(seed),
            rng_state=np.frombuffer(rng_state_after_x, dtype=np.uint8),
            transform_class=name,
            shape=np.asarray(shape, dtype=np.int64),
            kornia_version=getattr(kornia, "__version__", "unknown"),
        )
        return ("OK", str(out_path))
    except Exception as e:
        return ("SKIP", f"{name}__seed{seed}: {type(e).__name__}: {e}")


def main() -> None:
    transforms = discover()
    print(f"Discovered {len(transforms)} transforms")
    ok = 0
    skip = 0
    skip_reasons: list[str] = []
    for name, cls in transforms:
        for seed in SEEDS:
            for shape in SHAPES:
                status, info = generate_one(name, cls, seed, shape)
                if status == "OK":
                    ok += 1
                    print(f"  OK  {Path(info).name}")
                else:
                    skip += 1
                    skip_reasons.append(info)
                    print(f"  SKIP  {info}")

    print(f"\n{ok} goldens generated, {skip} skipped")
    if skip:
        print("Skipped (first 10):")
        for r in skip_reasons[:10]:
            print(f"  {r}")

    # Write index
    idx_path = DATA_DIR / "_index.json"
    files = sorted(p.name for p in DATA_DIR.glob("*.npz"))
    json.dump({"count": len(files), "files": files}, idx_path.open("w"), indent=2)
    print(f"\nIndex written: {idx_path}  ({len(files)} files)")


if __name__ == "__main__":
    main()
