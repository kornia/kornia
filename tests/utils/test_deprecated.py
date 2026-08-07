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

import warnings

import numpy as np
import pytest
import torch

from kornia import utils


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
        with pytest.warns(DeprecationWarning):
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
