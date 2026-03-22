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
