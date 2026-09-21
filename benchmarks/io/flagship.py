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

"""Flagship io benchmark: kornia.io image decoding vs OpenCV, torchvision and PIL.

Times what a data loader pays per image: read an encoded file from disk and decode it to an RGB
``uint8`` array or tensor. One row per common format:

=============  =====================  ==================================  =====================================
kornia.io      OpenCV                 torchvision.io                      PIL
=============  =====================  ==================================  =====================================
``load_jpeg``  ``imread`` + BGR->RGB  ``decode_image(read_file(path))``   ``Image.open(path).convert("RGB")``
``load_png``   ``imread`` + BGR->RGB  ``decode_image(read_file(path))``   ``Image.open(path).convert("RGB")``
=============  =====================  ==================================  =====================================

kornia calls ``load_image(path, ImageLoadType.RGB8, device)``, which decodes with kornia-rs and
returns a ``(3, H, W)`` uint8 tensor on the selected device; torchvision's tensor is moved to the
same device, so on an accelerator both columns include the host-to-device copy and the device sync.
OpenCV and PIL return host arrays, their native result, and never touch the accelerator. OpenCV's
row includes the ``cvtColor`` to RGB so every column produces the same channel order; PIL's
includes ``np.asarray``, which forces the lazy decode. Each call loads ``batch`` distinct files,
so the throughput is decoded images per second. The files are ``--size`` square RGB images of
Gaussian-blurred noise — compressible like a photograph rather than like white noise — written
once per config to a temporary directory (JPEG quality 90 via PIL, PNG default compression) and
read back through the OS page cache, so the rows measure decoding, not storage. There is no
``torch.compile`` column: file I/O and the kornia-rs decoder are outside what it compiles, so
``--compile`` is ignored.

Usage:
    python benchmarks/io/flagship.py --batches 1,8,32 --size 256 --device cpu
    python benchmarks/io/flagship.py --device cuda --json io_cuda.json
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import torch

# Prefer this checkout to an installed wheel or another editable checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    Backend,
    add_flagship_args,
    batch_list,
    finish_run,
    image_row_fields,
    optional_import,
    run_batch_sweep,
    setup_run,
    start_run,
)

from kornia.filters import gaussian_blur2d
from kornia.io import ImageLoadType, load_image

OPS = ("load_jpeg", "load_png")
FORMATS = {"load_jpeg": ("jpg", {"quality": 90}), "load_png": ("png", {})}


def write_images(root: Path, b: int, size: int, pil: ModuleType) -> dict[str, list[str]]:
    """Write ``b`` seeded smooth RGB images per format; returns ``{op: [paths]}``."""
    gen = torch.Generator().manual_seed(0)
    noise = torch.rand(b, 3, size, size, generator=gen)
    smooth = gaussian_blur2d(noise, (13, 13), (3.0, 3.0))
    lo, hi = smooth.amin(dim=(-2, -1), keepdim=True), smooth.amax(dim=(-2, -1), keepdim=True)
    arrays = ((smooth - lo) / (hi - lo) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    paths: dict[str, list[str]] = {}
    for op, (ext, options) in FORMATS.items():
        paths[op] = []
        for i, arr in enumerate(arrays):
            path = root / f"b{b}_{i}.{ext}"
            pil.fromarray(arr).save(path, **options)
            paths[op].append(str(path))
    return paths


def build_ops(
    b: int,
    size: int,
    root: Path,
    device: torch.device,
    cv2: Optional[ModuleType],
    tvio: Optional[ModuleType],
    pil: ModuleType,
    selected: Optional[frozenset[str]] = None,
) -> tuple[dict[str, dict[str, Backend]], dict[str, str]]:
    """Build {op: {backend: zero-arg callable}}; each callable loads ``b`` files once."""
    paths = write_images(root, b, size, pil)
    ops: dict[str, dict[str, Backend]] = {}
    for op in OPS:
        if selected is not None and op not in selected:
            continue
        files = paths[op]
        row: dict[str, Backend] = {
            "kornia (eager)": lambda files=files: [load_image(p, ImageLoadType.RGB8, device) for p in files]
        }
        if tvio:
            row["torchvision"] = lambda files=files: [
                tvio.decode_image(tvio.read_file(p), mode=tvio.ImageReadMode.RGB).to(device) for p in files
            ]
        if cv2:
            row["opencv"] = lambda files=files: [
                cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in files
            ]
        row["PIL"] = lambda files=files: [np.asarray(pil.open(p).convert("RGB")) for p in files]
        ops[op] = row
    return ops, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_flagship_args(parser, ops=OPS)
    args = parser.parse_args()
    device, _, sync = setup_run(args)
    args.dtype = "uint8"  # every decoder returns uint8; the header and the rows say so

    cv2, cv2_error = optional_import("cv2")
    tvio, tvio_error = optional_import("torchvision.io")
    pil, pil_error = optional_import("PIL.Image")
    if pil is None:
        raise SystemExit(f"PIL is required to write the benchmark's input files ({pil_error})")

    libs = [(tvio, "torchvision", tvio_error), (cv2, "opencv", cv2_error)]
    meta = start_run(
        "flagship io",
        args,
        device,
        units="img/s",
        regimes=[
            "kornia/torchvision: file -> uint8 CHW tensor on the device; opencv/PIL: file -> uint8 HWC host array",
            "--dtype does not apply (decoders return uint8); no torch.compile column",
        ],
        missing=[(name, err) for lib, name, err in libs if lib is None],
    )
    backends = ["kornia (eager)", "torchvision", "opencv", "PIL"]
    with tempfile.TemporaryDirectory(prefix="kornia-io-bench-") as tmp:
        results = run_batch_sweep(
            batch_list(args),
            lambda b: build_ops(b, args.size, Path(tmp), device, cv2, tvio, pil, args.ops),
            backends,
            row_fields=image_row_fields(args),
            sync=sync,
            units="img/s",
            min_run_time=args.min_run_time,
        )
    finish_run(args, "io", meta, results)


if __name__ == "__main__":
    main()
