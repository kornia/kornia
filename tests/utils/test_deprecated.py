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

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import io
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

import kornia
import kornia.core.ops
import kornia.core.utils
import kornia.geometry
import kornia.image
import kornia.io
import kornia.losses
from kornia import utils

from testing.base import BaseTester


class TestDeprecatedWrappers:
    """Verify that every re-exported function in kornia.utils emits a DeprecationWarning."""

    def test_create_meshgrid_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.geometry.create_meshgrid"):
            out = utils.create_meshgrid(4, 4)
        assert out.shape == (1, 4, 4, 2)

    def test_create_meshgrid3d_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.geometry.create_meshgrid3d"):
            out = utils.create_meshgrid3d(2, 3, 4)
        assert out.shape == (1, 2, 3, 4, 3)

    def test_draw_line_warns(self):
        img = torch.zeros(3, 16, 16)
        p1 = torch.tensor([0, 0])
        p2 = torch.tensor([7, 7])
        color = torch.tensor([1.0, 0.0, 0.0])
        with pytest.warns(DeprecationWarning, match="kornia.image.draw_line"):
            out = utils.draw_line(img, p1, p2, color)
        assert out.shape == img.shape

    def test_draw_rectangle_warns(self):
        img = torch.zeros(1, 3, 16, 16)
        rect = torch.tensor([[[2, 2, 10, 10]]], dtype=torch.float32)
        with pytest.warns(DeprecationWarning, match="kornia.image.draw_rectangle"):
            out = utils.draw_rectangle(img, rect)
        assert out.shape == img.shape

    def test_draw_point2d_warns(self):
        img = torch.zeros(1, 3, 16, 16)
        points = torch.tensor([[4, 4]], dtype=torch.long)
        color = torch.tensor([0.0, 1.0, 0.0])
        with pytest.warns(DeprecationWarning, match="kornia.image.draw_point2d"):
            out = utils.draw_point2d(img[0], points, color)
        assert out.shape == img[0].shape

    def test_draw_convex_polygon_warns(self):
        img = torch.zeros(1, 3, 16, 16)
        polygon = torch.tensor([[[2.0, 2.0], [14.0, 2.0], [14.0, 14.0], [2.0, 14.0]]])
        color = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.warns(DeprecationWarning, match="kornia.image.draw_convex_polygon"):
            out = utils.draw_convex_polygon(img, polygon, color)
        assert out.shape == img.shape

    def test_image_to_tensor_warns(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        with pytest.warns(DeprecationWarning, match="kornia.image.image_to_tensor"):
            t = utils.image_to_tensor(arr)
        assert isinstance(t, torch.Tensor)

    def test_tensor_to_image_warns(self):
        t = torch.zeros(3, 8, 8)
        with pytest.warns(DeprecationWarning, match="kornia.image.tensor_to_image"):
            arr = utils.tensor_to_image(t)
        assert arr is not None

    def test_one_hot_warns(self):
        labels = torch.tensor([0, 1, 2])
        with pytest.warns(DeprecationWarning, match="kornia.losses.one_hot"):
            out = utils.one_hot(labels, num_classes=3, device=torch.device("cpu"), dtype=torch.float32)
        assert out.shape == (3, 3)

    def test_all_wrappers_in_all(self):
        expected = {
            "create_meshgrid",
            "create_meshgrid3d",
            "draw_convex_polygon",
            "draw_line",
            "draw_point2d",
            "draw_rectangle",
            "image_to_string",
            "image_to_tensor",
            "load_pointcloud_ply",
            "one_hot",
            "print_image",
            "save_pointcloud_ply",
            "tensor_to_image",
        }
        assert expected.issubset(set(utils.__all__))

    def test_multiple_calls_each_warn(self):
        """Deprecation warning must be raised every time, not just the first call."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utils.create_meshgrid(2, 2)
            utils.create_meshgrid(2, 2)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) == 2

    def test_warning_can_be_escalated_to_an_error(self):
        """`-W error::DeprecationWarning` must turn the warning into an error.

        See https://github.com/kornia/kornia/issues/3956: the emitter used to prepend its
        own "always" filter, which shadowed the caller's "error" filter and exempted kornia
        from a documented CPython mechanism.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(DeprecationWarning):
                utils.create_meshgrid(2, 2)

    def test_warning_does_not_mutate_the_global_filters(self):
        """Emitting the warning must leave `warnings.filters` exactly as it found it.

        The two `simplefilter` calls used to prepend a global entry each, so a single call
        to any deprecated symbol permanently rewrote the process-wide warning config.
        """
        with warnings.catch_warnings():
            warnings.resetwarnings()
            assert warnings.filters == []
            utils.create_meshgrid(2, 2)
            assert warnings.filters == []


class TestRestoredWrappers:
    """Names removed in 0.8.3 without a deprecation window, restored as warning shims.

    See https://github.com/kornia/kornia/pull/3891 follow-up audit: of the 34 public
    `kornia.utils` names in v0.8.2, 20 were removed with no shim. These are the ones
    whose implementations still exist elsewhere in the library.
    """

    def test_eye_like_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.core.ops.eye_like"):
            out = utils.eye_like(3, torch.rand(2, 4, 4))
        assert out.shape == (2, 3, 3)

    def test_vec_like_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.core.ops.vec_like"):
            out = utils.vec_like(3, torch.rand(2, 4, 4))
        assert out.shape == (2, 3, 1)

    def test_safe_solve_with_mask_warns(self):
        A = torch.eye(3)[None]
        b = torch.rand(1, 3, 3)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.safe_solve_with_mask"):
            sol, _, mask = utils.safe_solve_with_mask(b, A)
        assert sol.shape == b.shape
        assert bool(mask.all())

    def test_safe_inverse_with_mask_warns(self):
        A = torch.eye(3)[None]
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.safe_inverse_with_mask"):
            inv, mask = utils.safe_inverse_with_mask(A)
        assert inv.shape == A.shape
        assert bool(mask.all())

    def test_is_mps_tensor_safe_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.is_mps_tensor_safe"):
            out = utils.is_mps_tensor_safe(torch.rand(1))
        assert out is False

    def test_dataclass_roundtrip_warns(self):
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            a: int

        with pytest.warns(DeprecationWarning, match="kornia.core.utils.dataclass_to_dict"):
            d = utils.dataclass_to_dict(_Cfg(a=1))
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.dict_to_dataclass"):
            cfg = utils.dict_to_dataclass(d, _Cfg)
        assert cfg == _Cfg(a=1)

    def test_image_list_to_tensor_warns(self):
        arrs = [np.zeros((8, 8, 3), dtype=np.uint8)] * 2
        with pytest.warns(DeprecationWarning, match="kornia.image.image_list_to_tensor"):
            t = utils.image_list_to_tensor(arrs)
        assert t.shape == (2, 3, 8, 8)

    def test_image_to_tensor_module_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.image.ImageToTensor"):
            mod = utils.ImageToTensor()
        out = mod(np.zeros((8, 8, 3), dtype=np.uint8))
        assert isinstance(out, torch.Tensor)

    def test_image_to_tensor_is_a_real_class(self):
        """The shim must stay a type: isinstance checks and subclassing worked in 0.8.2."""
        from kornia.image import ImageToTensor as NewImageToTensor

        assert isinstance(utils.ImageToTensor, type)
        assert issubclass(utils.ImageToTensor, NewImageToTensor)
        with pytest.warns(DeprecationWarning, match=r"kornia\.image\.ImageToTensor"):
            inst = utils.ImageToTensor()
        assert isinstance(inst, NewImageToTensor)

    def test_cached_downloader_warns(self):
        with pytest.warns(DeprecationWarning, match="kornia.onnx.download.CachedDownloader"):
            cls = utils.CachedDownloader
        from kornia.onnx.download import CachedDownloader

        assert cls is CachedDownloader

    def test_batched_forward_warns(self):
        x = torch.rand(300, 2)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.batched_forward"):
            out = utils.batched_forward(torch.nn.Identity(), x, torch.device("cpu"), batch_size=128)
        assert torch.allclose(out, x)

    def test_get_sample_images_no_download_warns(self, tmp_path):
        # empty path list: warns and returns an empty list without touching the network
        with pytest.warns(DeprecationWarning, match="kornia.io.get_sample_images"):
            out = utils.get_sample_images(paths=[], download=False, cache_dir=str(tmp_path))
        assert out == []

    def test_torch_meshgrid_warns(self):
        with pytest.warns(DeprecationWarning, match="torch.meshgrid"):
            xs, _ys = utils.torch_meshgrid([torch.arange(2), torch.arange(3)], indexing="ij")
        assert xs.shape == (2, 3)

    def test_map_location_to_cpu_warns(self):
        t = torch.rand(2)
        with pytest.warns(DeprecationWarning, match="map_location"):
            out = utils.map_location_to_cpu(t, "cuda:0")
        assert out is t

    def test_device_and_env_helpers_warn(self):
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.get_cuda_device_if_available"):
            dev = utils.get_cuda_device_if_available()
        assert isinstance(dev, torch.device)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.get_mps_device_if_available"):
            dev = utils.get_mps_device_if_available()
        assert isinstance(dev, torch.device)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.get_cuda_or_mps_device_if_available"):
            dev = utils.get_cuda_or_mps_device_if_available()
        assert isinstance(dev, torch.device)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.is_autocast_enabled"):
            out = utils.is_autocast_enabled()
        assert isinstance(out, bool)
        with pytest.warns(DeprecationWarning, match="kornia.core.utils.xla_is_available"):
            out = utils.xla_is_available()
        assert isinstance(out, bool)

    def test_v082_public_surface_fully_restored(self):
        """Every public name from v0.8.2's kornia.utils.__all__ must still resolve.

        Frozen copy of `git show v0.8.2:kornia/utils/__init__.py` __all__, minus the
        underscore-prefixed `_extract_device_dtype` (private per the stability policy).
        This is the closure of the 0.8.3 hard-removal audit: if any name here stops
        resolving, the deprecation window was skipped again.
        """
        v082_public_all = [
            "CachedDownloader",
            "ImageToTensor",
            "batched_forward",
            "create_meshgrid",
            "create_meshgrid3d",
            "dataclass_to_dict",
            "deprecated",
            "dict_to_dataclass",
            "draw_convex_polygon",
            "draw_line",
            "draw_point2d",
            "draw_rectangle",
            "eye_like",
            "get_cuda_device_if_available",
            "get_cuda_or_mps_device_if_available",
            "get_mps_device_if_available",
            "get_sample_images",
            "image_list_to_tensor",
            "image_to_string",
            "image_to_tensor",
            "is_autocast_enabled",
            "is_mps_tensor_safe",
            "load_pointcloud_ply",
            "map_location_to_cpu",
            "one_hot",
            "print_image",
            "safe_inverse_with_mask",
            "safe_solve_with_mask",
            "save_pointcloud_ply",
            "tensor_to_image",
            "torch_meshgrid",
            "vec_like",
            "xla_is_available",
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unresolvable = [n for n in v082_public_all if not hasattr(utils, n)]
        assert not unresolvable, f"v0.8.2 public names no longer resolvable: {unresolvable}"

    def test_restored_wrappers_in_all(self):
        expected = {
            "CachedDownloader",
            "ImageToTensor",
            "batched_forward",
            "dataclass_to_dict",
            "dict_to_dataclass",
            "eye_like",
            "get_cuda_device_if_available",
            "get_cuda_or_mps_device_if_available",
            "get_mps_device_if_available",
            "get_sample_images",
            "image_list_to_tensor",
            "is_autocast_enabled",
            "is_mps_tensor_safe",
            "map_location_to_cpu",
            "safe_inverse_with_mask",
            "safe_solve_with_mask",
            "torch_meshgrid",
            "vec_like",
            "xla_is_available",
        }
        assert expected.issubset(set(utils.__all__))


# The shims stand in for the 0.8.2 functions on every surface those supported: attribute access after a plain
# ``import kornia`` (#5900), the full signature, TorchScript and ``torch.compile(fullgraph=True)`` (#5901), and a
# visible warning when ``torch.load`` calls ``map_location_to_cpu`` (#5905).


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports the kornia tree under test.

    The child runs from the directory holding that ``kornia`` package, so ``-c`` resolves it ahead of any installed
    copy, and without ``PYTHONWARNINGS``, so it starts from Python's default warning filters.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    # Trusted, fixed command: the current interpreter running a literal script; no external input.
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(kornia.__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _same_kornia(path: str) -> bool:
    return os.path.realpath(path) == os.path.realpath(kornia.__file__)


class TestReachableAfterBareImport:
    """``import kornia`` alone reaches the shims, the way kornia's 0.8.2 documentation used them (#5900)."""

    def test_attribute_access_after_import_kornia(self):
        result = _run_fresh(
            """
            import sys
            import warnings

            import torch  # torch's own import-time warnings are not kornia's

            warnings.simplefilter("error")
            import kornia

            print(kornia.__file__)
            print("kornia.utils imported by import kornia:", "kornia.utils" in sys.modules)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                grid = kornia.utils.create_meshgrid(2, 3); line = sys._getframe().f_lineno
            print("shape:", tuple(grid.shape))
            print("warnings:", [(w.category.__name__, w.lineno == line) for w in caught])
            """
        )
        assert result.returncode == 0, result.stderr
        first, *rest = result.stdout.splitlines()
        assert _same_kornia(first)
        assert rest == [
            "kornia.utils imported by import kornia: False",
            "shape: (1, 2, 3, 2)",
            "warnings: [('DeprecationWarning', True)]",
        ]

    def test_v082_sample_submodule_after_import_kornia(self, tmp_path):
        result = _run_fresh(
            f"""
            import warnings

            import kornia

            print(kornia.__file__)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fn = kornia.utils.sample.get_sample_images
                print("warnings on access:", len(caught))
                print("result:", fn(paths=[], download=False, cache_dir={str(tmp_path)!r}))
            names_replacement = ["`kornia.io.get_sample_images`" in str(w.message) for w in caught]
            print("warnings:", list(zip([w.category.__name__ for w in caught], names_replacement)))
            """
        )
        assert result.returncode == 0, result.stderr
        first, *rest = result.stdout.splitlines()
        assert _same_kornia(first)
        assert rest == ["warnings on access: 0", "result: []", "warnings: [('DeprecationWarning', True)]"]

    def test_v082_sample_submodule_import(self, tmp_path):
        from kornia.utils import sample
        from kornia.utils.sample import get_sample_images

        assert sample.__all__ == ["get_sample_images"]
        with pytest.warns(DeprecationWarning, match=r"`kornia\.io\.get_sample_images`"):
            assert get_sample_images(paths=[], download=False, cache_dir=str(tmp_path)) == []


@dataclasses.dataclass
class _Cfg:
    a: int
    b: float = 2.0


class _Scale(torch.nn.Module):
    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return x * scale


# Each shim's name -> the replacement its warning names, which is also the function it forwards to.
_UTILS_REPLACEMENTS = {
    "batched_forward": "kornia.core.utils.batched_forward",
    "create_meshgrid": "kornia.geometry.create_meshgrid",
    "create_meshgrid3d": "kornia.geometry.create_meshgrid3d",
    "dataclass_to_dict": "kornia.core.utils.dataclass_to_dict",
    "dict_to_dataclass": "kornia.core.utils.dict_to_dataclass",
    "draw_convex_polygon": "kornia.image.draw_convex_polygon",
    "draw_line": "kornia.image.draw_line",
    "draw_point2d": "kornia.image.draw_point2d",
    "draw_rectangle": "kornia.image.draw_rectangle",
    "eye_like": "kornia.core.ops.eye_like",
    "get_cuda_device_if_available": "kornia.core.utils.get_cuda_device_if_available",
    "get_cuda_or_mps_device_if_available": "kornia.core.utils.get_cuda_or_mps_device_if_available",
    "get_mps_device_if_available": "kornia.core.utils.get_mps_device_if_available",
    "get_sample_images": "kornia.io.get_sample_images",
    "image_list_to_tensor": "kornia.image.image_list_to_tensor",
    "image_to_string": "kornia.image.image_to_string",
    "image_to_tensor": "kornia.image.image_to_tensor",
    "is_autocast_enabled": "kornia.core.utils.is_autocast_enabled",
    "is_mps_tensor_safe": "kornia.core.utils.is_mps_tensor_safe",
    "load_pointcloud_ply": "kornia.geometry.load_pointcloud_ply",
    "map_location_to_cpu": 'torch.load(..., map_location="cpu")',
    "one_hot": "kornia.losses.one_hot",
    "print_image": "kornia.image.print_image",
    "safe_inverse_with_mask": "kornia.core.utils.safe_inverse_with_mask",
    "safe_solve_with_mask": "kornia.core.utils.safe_solve_with_mask",
    "save_pointcloud_ply": "kornia.geometry.save_pointcloud_ply",
    "tensor_to_image": "kornia.image.tensor_to_image",
    "torch_meshgrid": "torch.meshgrid",
    "vec_like": "kornia.core.ops.vec_like",
    "xla_is_available": "kornia.core.utils.xla_is_available",
}

# The deprecated top-level aliases in ``kornia/__init__.py`` -> their replacements.
_TOP_LEVEL_REPLACEMENTS = {
    name: _UTILS_REPLACEMENTS[name]
    for name in (
        "create_meshgrid",
        "create_meshgrid3d",
        "draw_convex_polygon",
        "draw_line",
        "draw_point2d",
        "draw_rectangle",
        "eye_like",
        "get_cuda_device_if_available",
        "get_cuda_or_mps_device_if_available",
        "get_mps_device_if_available",
        "image_to_string",
        "image_to_tensor",
        "is_autocast_enabled",
        "load_pointcloud_ply",
        "one_hot",
        "print_image",
        "save_pointcloud_ply",
        "tensor_to_image",
        "vec_like",
        "xla_is_available",
    )
}

# Shims that forward to a kornia function of the same signature (the other two keep their 0.8.2 signatures).
_FORWARDING = sorted(set(_UTILS_REPLACEMENTS) - {"map_location_to_cpu", "torch_meshgrid"})


def _resolve(dotted: str) -> Any:
    obj: Any = kornia
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def _target(name: str) -> Callable[..., Any]:
    if name == "torch_meshgrid":
        return lambda tensors, indexing: torch.meshgrid(tensors, indexing=indexing)
    if name == "map_location_to_cpu":
        return lambda storage, location: storage
    return _resolve(_UTILS_REPLACEMENTS[name])


def _call_args(name: str, device: torch.device, dtype: torch.dtype, tmp_path: Path) -> tuple[tuple, dict]:
    """Fresh arguments for one call of ``name``, giving every parameter that has a default a non-default value."""
    img = torch.zeros(1, 3, 6, 7, device=device, dtype=dtype)
    color = torch.tensor([1.0, 0.5, 0.25], device=device, dtype=dtype)
    x = torch.eye(2, device=device, dtype=dtype) * 2 + 0.25
    arr = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    calls: dict[str, Callable[[], tuple[tuple, dict]]] = {
        "batched_forward": lambda: ((_Scale(), x[None].repeat(3, 1, 1), device, 2), {"scale": 3.0}),
        # align_corners only matters for normalized coordinates, so those stay on.
        "create_meshgrid": lambda: ((3, 4, True, device, dtype, False), {}),
        "create_meshgrid3d": lambda: ((2, 3, 4, False, device, dtype), {}),
        "dataclass_to_dict": lambda: ((_Cfg(1, 3.0),), {}),
        "dict_to_dataclass": lambda: (({"a": 1, "b": 3.0}, _Cfg), {}),
        "draw_convex_polygon": lambda: (
            (img, torch.tensor([[[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]]], device=device, dtype=dtype), color),
            {},
        ),
        "draw_line": lambda: (
            (img[0], torch.tensor([0, 0], device=device), torch.tensor([6, 5], device=device), color),
            {},
        ),
        "draw_point2d": lambda: ((img[0], torch.tensor([[1, 2], [4, 3]], device=device), color), {}),
        "draw_rectangle": lambda: ((img, torch.tensor([[[1, 1, 5, 4]]], device=device), color, None, 2), {}),
        "eye_like": lambda: ((3, x[None].repeat(2, 1, 1), True), {}),
        "get_cuda_device_if_available": lambda: ((0,), {}),
        "get_cuda_or_mps_device_if_available": lambda: ((), {}),
        "get_mps_device_if_available": lambda: ((), {}),
        "get_sample_images": lambda: ((None, [], False, str(tmp_path), True, None), {}),
        "image_list_to_tensor": lambda: (([arr, arr + 1],), {}),
        "image_to_string": lambda: ((img[0] + 0.5, 4), {}),
        "image_to_tensor": lambda: ((arr, False), {}),
        "is_autocast_enabled": lambda: ((False,), {}),
        "is_mps_tensor_safe": lambda: ((x,), {}),
        "load_pointcloud_ply": lambda: ((tmp_path / "in.ply", None), {}),
        "map_location_to_cpu": lambda: ((x, "cuda:0"), {}),
        "one_hot": lambda: ((torch.tensor([[0, 2], [1, 0]], device=device), 3, device, dtype, 0.25), {}),
        "print_image": lambda: ((img[0] + 0.5, 4), {}),
        "safe_inverse_with_mask": lambda: ((x[None],), {}),
        "safe_solve_with_mask": lambda: ((x[None], x[None] + 1), {}),
        "save_pointcloud_ply": lambda: ((tmp_path / "out.ply", torch.arange(9.0).view(3, 3)), {}),
        "tensor_to_image": lambda: ((img + 0.5, True, True), {}),
        "torch_meshgrid": lambda: (([torch.arange(2, device=device), torch.arange(3, device=device)], "xy"), {}),
        "vec_like": lambda: ((3, x[None].repeat(2, 1, 1), True), {}),
        "xla_is_available": lambda: ((), {}),
    }
    if name == "load_pointcloud_ply":
        kornia.geometry.save_pointcloud_ply(tmp_path / "in.ply", torch.arange(9.0).view(3, 3) / 7)
    return calls[name]()


def _outcome(fn: Callable[..., Any], args: tuple, kwargs: dict) -> tuple:
    """What calling ``fn`` gives: ``("returned", value, stdout)`` or ``("raised", type, message)``."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            value = fn(*args, **kwargs)
    except Exception as e:  # the comparison is between two calls, whatever they raise
        return ("raised", type(e), str(e))
    return ("returned", value, out.getvalue())


def _assert_same(got: Any, expected: Any) -> None:
    assert type(got) is type(expected)
    if isinstance(got, torch.Tensor):
        assert (got.dtype, got.device, got.shape) == (expected.dtype, expected.device, expected.shape)
        assert got.stride() == expected.stride()
        assert torch.equal(got, expected)
    elif isinstance(got, np.ndarray):
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)
    elif isinstance(got, (tuple, list)):
        assert len(got) == len(expected)
        for g, e in zip(got, expected):
            _assert_same(g, e)
    else:
        assert got == expected


def _signature(fn: Callable[..., Any]) -> str:
    return str(inspect.signature(fn, eval_str=True))


class TestShimsForwardEveryParameter(BaseTester):
    """Every shim forwards each parameter to its replacement and warns once, at the caller's line (#5901)."""

    @pytest.mark.parametrize("name", sorted(_UTILS_REPLACEMENTS))
    def test_utils_shim_matches_its_replacement(self, name, device, dtype, tmp_path):
        expected = _outcome(_target(name), *_call_args(name, device, dtype, tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _outcome(getattr(utils, name), *_call_args(name, device, dtype, tmp_path))
        _assert_same(got, expected)
        (w,) = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert w.filename == __file__
        assert f"`{_UTILS_REPLACEMENTS[name]}`" in str(w.message)
        # kornia.utils is deprecated, not removed: every name in it still works.
        assert "removed" not in str(w.message)

    @pytest.mark.parametrize("name", sorted(_TOP_LEVEL_REPLACEMENTS))
    def test_top_level_alias_matches_its_replacement(self, name, device, dtype, tmp_path):
        expected = _outcome(_target(name), *_call_args(name, device, dtype, tmp_path))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = _outcome(getattr(kornia, name), *_call_args(name, device, dtype, tmp_path))
        _assert_same(got, expected)
        (w,) = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert w.filename == __file__
        assert f"`{_TOP_LEVEL_REPLACEMENTS[name]}`" in str(w.message)

    @pytest.mark.parametrize("name", _FORWARDING)
    def test_utils_shim_has_its_replacements_signature(self, name):
        assert _signature(getattr(utils, name)) == _signature(_target(name))

    @pytest.mark.parametrize("name", sorted(_TOP_LEVEL_REPLACEMENTS))
    def test_top_level_alias_has_its_replacements_signature(self, name):
        assert _signature(getattr(kornia, name)) == _signature(_target(name))

    def test_v082_signatures_of_the_other_two(self):
        assert _signature(utils.torch_meshgrid) == "(tensors: List[torch.Tensor], indexing: str)"
        assert _signature(utils.map_location_to_cpu) == "(storage: Any, *args: Any, **kwargs: Any) -> Any"

    def test_image_to_tensor_class_has_its_replacements_signature(self):
        assert _signature(utils.ImageToTensor) == _signature(kornia.image.ImageToTensor)

    def test_warning_points_at_the_callers_line(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            line = sys._getframe().f_lineno
            utils.create_meshgrid(2, 2)
            kornia.create_meshgrid(2, 2)
            utils.ImageToTensor()
            utils.CachedDownloader  # noqa: B018  the attribute access is what warns
        assert [(w.category, w.filename, w.lineno) for w in caught] == [
            (DeprecationWarning, __file__, line + offset) for offset in (1, 2, 3, 4)
        ]


# User code that calls a shim, as 0.8.2 code did. Each takes and returns one tensor so that one test scripts or
# compiles all of them. Before #3449 the six ``kornia.utils`` calls among them scripted (#5901).


def _call_utils_create_meshgrid(x: torch.Tensor) -> torch.Tensor:
    return utils.create_meshgrid(3, 4, False, x.device, x.dtype)


def _call_utils_create_meshgrid3d(x: torch.Tensor) -> torch.Tensor:
    return utils.create_meshgrid3d(2, 3, 4, False, x.device, x.dtype)


def _call_utils_one_hot(x: torch.Tensor) -> torch.Tensor:
    return utils.one_hot(torch.tensor([[0, 2], [1, 0]], device=x.device), 3, x.device, x.dtype)


def _call_utils_draw_rectangle(x: torch.Tensor) -> torch.Tensor:
    image = torch.zeros(1, 3, 6, 7, device=x.device, dtype=x.dtype)
    return utils.draw_rectangle(image, torch.tensor([[[1, 1, 5, 4]]], device=x.device))


def _call_utils_torch_meshgrid(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(utils.torch_meshgrid([x[0, 0], x[0, 1, :2]], "ij"))


def _call_utils_safe_inverse_with_mask(x: torch.Tensor) -> torch.Tensor:
    inverse, mask = utils.safe_inverse_with_mask(x)
    return torch.cat([inverse.flatten(), mask.to(inverse.dtype)])


def _call_utils_eye_like(x: torch.Tensor) -> torch.Tensor:
    return utils.eye_like(3, x)


def _call_utils_vec_like(x: torch.Tensor) -> torch.Tensor:
    return utils.vec_like(3, x)


def _call_utils_draw_point2d(x: torch.Tensor) -> torch.Tensor:
    image = torch.zeros(3, 6, 7, device=x.device, dtype=x.dtype)
    color = torch.ones(3, device=x.device, dtype=x.dtype)
    return utils.draw_point2d(image, torch.tensor([[1, 2], [4, 3]], device=x.device), color)


def _call_create_meshgrid(x: torch.Tensor) -> torch.Tensor:
    return kornia.create_meshgrid(3, 4, False, x.device, x.dtype)


def _call_create_meshgrid3d(x: torch.Tensor) -> torch.Tensor:
    return kornia.create_meshgrid3d(2, 3, 4, False, x.device, x.dtype)


def _call_one_hot(x: torch.Tensor) -> torch.Tensor:
    return kornia.one_hot(torch.tensor([[0, 2], [1, 0]], device=x.device), 3, x.device, x.dtype)


def _call_draw_rectangle(x: torch.Tensor) -> torch.Tensor:
    image = torch.zeros(1, 3, 6, 7, device=x.device, dtype=x.dtype)
    return kornia.draw_rectangle(image, torch.tensor([[[1, 1, 5, 4]]], device=x.device))


def _call_eye_like(x: torch.Tensor) -> torch.Tensor:
    return kornia.eye_like(3, x)


def _call_vec_like(x: torch.Tensor) -> torch.Tensor:
    return kornia.vec_like(3, x)


def _call_draw_point2d(x: torch.Tensor) -> torch.Tensor:
    image = torch.zeros(3, 6, 7, device=x.device, dtype=x.dtype)
    color = torch.ones(3, device=x.device, dtype=x.dtype)
    return kornia.draw_point2d(image, torch.tensor([[1, 2], [4, 3]], device=x.device), color)


_SCRIPTABLE_CALLERS = [
    _call_utils_create_meshgrid,
    _call_utils_create_meshgrid3d,
    _call_utils_one_hot,
    _call_utils_draw_rectangle,
    _call_utils_torch_meshgrid,
    _call_utils_safe_inverse_with_mask,
    _call_utils_eye_like,
    _call_utils_vec_like,
    _call_utils_draw_point2d,
    _call_create_meshgrid,
    _call_create_meshgrid3d,
    _call_one_hot,
    _call_draw_rectangle,
    _call_eye_like,
    _call_vec_like,
    _call_draw_point2d,
]

# The callers whose replacement compiles with ``fullgraph=True`` (the draw functions read tensor values back).
_COMPILABLE_CALLERS = [
    _call_utils_create_meshgrid,
    _call_utils_create_meshgrid3d,
    _call_utils_one_hot,
    _call_utils_torch_meshgrid,
    _call_utils_safe_inverse_with_mask,
    _call_utils_eye_like,
    _call_utils_vec_like,
    _call_create_meshgrid,
    _call_one_hot,
]


def _invertible(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(3, device=device, dtype=dtype) * 2 + torch.arange(9, device=device, dtype=dtype).view(3, 3) / 16


class TestShimsUnderScriptAndCompile(BaseTester):
    """Code that calls a shim scripts and compiles as it did with the 0.8.2 function (#5901)."""

    @pytest.mark.parametrize("caller", _SCRIPTABLE_CALLERS, ids=lambda f: f.__name__[len("_call_") :])
    def test_jit_script_of_code_calling_a_shim(self, caller, device, dtype):
        x = _invertible(device, dtype)[None].repeat(2, 1, 1)
        with warnings.catch_warnings():
            # The shim's DeprecationWarning in eager, and torch.jit.script's own FutureWarning on torch >= 2.10.
            warnings.simplefilter("ignore")
            expected = _outcome(caller, (x,), {})
            if expected[0] == "raised":
                pytest.skip(f"the replacement itself raises here: {expected[1].__name__}: {expected[2]}")
            scripted = torch.jit.script(caller)
        self.assert_close(scripted(x), expected[1], rtol=0, atol=0)

    @pytest.mark.parametrize("caller", _COMPILABLE_CALLERS, ids=lambda f: f.__name__[len("_call_") :])
    def test_compile_fullgraph_of_code_calling_a_shim(self, caller, device, dtype, torch_optimizer):
        x = _invertible(device, dtype)[None].repeat(2, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = _outcome(caller, (x,), {})
        if expected[0] == "raised":
            pytest.skip(f"the replacement itself raises here: {expected[1].__name__}: {expected[2]}")
        compiled = torch_optimizer(caller, fullgraph=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = compiled(x)
        self.assert_close(got, expected[1])
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning) and "kornia" in str(w.message)]
        # Eager calls still warn, once each.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            caller(x)
        assert len([w for w in caught if issubclass(w.category, DeprecationWarning)]) == 1


def _checkpoint(n: int = 6) -> io.BytesIO:
    buf = io.BytesIO()
    torch.save({f"t{i}": torch.full((2,), float(i)) for i in range(n)}, buf)
    buf.seek(0)
    return buf


def _map_location_warnings(caught: list[warnings.WarningMessage]) -> list[tuple[str, int]]:
    return [
        (w.filename, w.lineno)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "`map_location_to_cpu`" in str(w.message)
    ]


class TestMapLocationToCpuUnderTorchLoad:
    """``torch.load`` calls the hook once per storage, from inside torch (#5905)."""

    @pytest.mark.parametrize("weights_only", [True, False])
    def test_one_warning_per_torch_load_at_the_callers_line(self, weights_only):
        buf = _checkpoint()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state = torch.load(buf, map_location=utils.map_location_to_cpu, weights_only=weights_only)
            line = sys._getframe().f_lineno - 1
        assert len(state) == 6
        assert _map_location_warnings(caught) == [(__file__, line)]

    def test_each_torch_load_warns(self):
        buf = _checkpoint()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                buf.seek(0)
                torch.load(buf, map_location=utils.map_location_to_cpu, weights_only=True)
        assert len(_map_location_warnings(caught)) == 3

    def test_shown_under_the_default_filters(self):
        result = _run_fresh(
            """
            import io

            import torch

            import kornia
            import kornia.utils

            print(kornia.__file__)
            buf = io.BytesIO()
            torch.save({f"t{i}": torch.full((2,), float(i)) for i in range(6)}, buf)
            buf.seek(0)
            state = torch.load(buf, map_location=kornia.utils.map_location_to_cpu, weights_only=True)
            print(len(state))
            """
        )
        assert result.returncode == 0, result.stderr
        assert _same_kornia(result.stdout.splitlines()[0])
        assert result.stdout.splitlines()[1] == "6"
        shown = [line for line in result.stderr.splitlines() if "DeprecationWarning" in line]
        assert len(shown) == 1, result.stderr
        assert shown[0].startswith("<string>:13: DeprecationWarning: ")
        assert "`map_location_to_cpu`" in shown[0]
