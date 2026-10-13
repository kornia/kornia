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

import io
import math
import warnings

import pytest
import torch

from kornia.core._compat import torch_version_ge
from kornia.core.exceptions import BaseError, ShapeError
from kornia.geometry import create_meshgrid
from kornia.image import draw_convex_polygon, draw_rectangle
from kornia.image.draw import draw_line, draw_point2d

from testing.base import BaseTester, dynamo_is_available


class TestDrawPoint(BaseTester):
    """Test drawing individual pixels."""

    def test_draw_point2d_rgb(self, dtype, device):
        """Test plotting multiple [x, y] points."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        color = torch.tensor([5, 10, 15], dtype=dtype, device=device)
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        img = draw_point2d(img, points, color)
        for x, y in points:
            self.assert_close(img[:, y, x], color.to(img.dtype))

    def test_draw_point2d_grayscale_third_order(self, dtype, device):
        """Test plotting multiple [x, y] points on a (1, m, n) image."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        color = torch.tensor([100], dtype=dtype, device=device)
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_point2d(img, points, color)
        for x, y in points:
            self.assert_close(img[:, y, x], color.to(img.dtype))

    def test_draw_point2d_grayscale_second_order(self, dtype, device):
        """Test plotting multiple [x, y] points on a (m, n) image."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        color = torch.tensor([100], dtype=dtype, device=device)
        img = torch.zeros(8, 8, dtype=dtype, device=device)
        img = draw_point2d(img, points, color)
        for x, y in points:
            self.assert_close(torch.unsqueeze(img[y, x], dim=0), color.to(img.dtype))

    def test_draw_point2d_with_mismatched_dims(self, dtype, device):
        """Test that we raise if the len of the color tensor != the # of image channels."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        color = torch.tensor([100, 50], dtype=dtype, device=device)
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        with pytest.raises(BaseError, match="Color dim must match"):
            draw_point2d(img, points, color)

    def test_draw_point2d_with_mismatched_dtype(self, device):
        """Test that the color is correctly cast to the image dtype when drawing points."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        color = torch.tensor([5, 10, 15], dtype=torch.float32, device=device)
        img = torch.zeros(3, 8, 8, dtype=torch.uint8, device=device)
        img = draw_point2d(img, points, color)
        assert img.dtype is torch.uint8
        for x, y in points:
            self.assert_close(img[:, y, x], color.to(torch.uint8))

    def test_draw_point2d_with_singleton_color_dims(self, dtype, device):
        """Ensure that plotting behavior is consistent if we have a singleton dim for the color."""
        points = torch.tensor([(1, 3), (2, 4)], device=device)
        # Plot given a color tensor of shape [3]
        color_vec = torch.tensor([5, 10, 15], dtype=torch.float32, device=device)
        vec_img = torch.zeros(3, 8, 8, dtype=torch.uint8, device=device)
        drawn_vec_img = draw_point2d(vec_img, points, color_vec)
        # Plot given a color tensor of shape [3, 1]
        color_mat = torch.unsqueeze(color_vec, dim=1)
        mat_img = vec_img.clone()
        drawn_mat_img = draw_point2d(mat_img, points, color_mat)
        # Ensure that we get the same underlying image back
        self.assert_close(drawn_vec_img, drawn_mat_img)

    def test_draw_point2d_single_1d_point(self, dtype, device):
        """A single [x, y] vector used to TypeError via zip over 0-d scalars."""
        points = torch.tensor([1, 3], device=device)
        color = torch.tensor([5, 10, 15], dtype=dtype, device=device)
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        img = draw_point2d(img, points, color)
        self.assert_close(img[:, 3, 1], color.to(img.dtype))

    def test_draw_point2d_empty_points(self, dtype, device):
        """An empty (0, 2) point set used to fail unpacking zip(*points)."""
        points = torch.zeros(0, 2, device=device)
        color = torch.tensor([5, 10, 15], dtype=dtype, device=device)
        img = torch.arange(3 * 8 * 8, device=device).reshape(3, 8, 8).to(dtype)
        expected = img.clone()
        out = draw_point2d(img, points, color)
        assert out is img
        self.assert_close(out, expected)

    @pytest.mark.parametrize("points", [[[1, 3], [2, 4]], [1, 3]])
    @pytest.mark.parametrize("shape", [(3, 8, 8), (8, 8)])
    def test_draw_point2d_draws_in_place_and_returns_the_input(self, points, shape, dtype, device):
        """The Return section documents in-place drawing into, and returning, the input tensor."""
        img = torch.zeros(shape, dtype=dtype, device=device)
        color = torch.ones(shape[0] if len(shape) == 3 else 1, dtype=dtype, device=device)
        out = draw_point2d(img, torch.tensor(points, device=device), color)
        assert out is img
        assert img.count_nonzero() > 0

    @pytest.mark.parametrize(
        "shape, match",
        [
            ((0,), "1D points tensor"),
            ((3,), "1D points tensor"),
            ((0, 3), "shape \\(N, 2\\)"),
            ((5, 0), "shape \\(N, 2\\)"),
            ((2, 3), "shape \\(N, 2\\)"),
            ((2, 2, 2), "shape \\(N, 2\\)"),
        ],
        ids=["empty-1d", "three-1d", "empty-three-columns", "zero-columns", "three-columns", "batched"],
    )
    def test_draw_point2d_rejects_malformed_points(self, shape, match, dtype, device):
        """Only (N, 2) and (2,) are accepted, including when the tensor is empty."""
        points = torch.zeros(shape, dtype=torch.int64, device=device)
        color = torch.tensor([5, 10, 15], dtype=dtype, device=device)
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        with pytest.raises(BaseError, match=match):
            draw_point2d(img, points, color)

    @pytest.mark.parametrize("shape", [(1, 4, 4), (4, 4)])
    def test_draw_point2d_accepts_0d_scalar_color(self, shape, dtype, device):
        """A 0-d color used to IndexError reading its channel dimension, as draw_line did before."""
        img = torch.zeros(shape, dtype=dtype, device=device)
        out = draw_point2d(img, torch.tensor([[1, 1]], device=device), torch.tensor(9, dtype=dtype, device=device))
        assert out[..., 1, 1].flatten().tolist() == [9.0]

    @pytest.mark.parametrize("color", [torch.tensor([7.0]), torch.tensor(7.0)])
    def test_draw_point2d_single_value_color_fills_every_channel(self, color, dtype, device):
        img = torch.zeros(3, 4, 5, dtype=dtype, device=device)
        out = draw_point2d(img, torch.tensor([[1, 2]], device=device), color.to(device))
        assert out[:, 2, 1].tolist() == [7.0, 7.0, 7.0]
        assert int(out.count_nonzero()) == 3

    def test_draw_point2d_rejects_a_batched_image(self, dtype, device):
        img = torch.zeros(1, 3, 4, 5, dtype=dtype, device=device)
        with pytest.raises(ValueError, match=r"\(C, H, W\)"):
            draw_point2d(img, torch.tensor([[1, 2]], device=device), torch.tensor([1.0], device=device))

    def test_draw_point2d_negative_coordinate_raises(self, dtype, device):
        img = torch.zeros(1, 3, 4, dtype=dtype, device=device)  # H = 3, W = 4
        with pytest.raises(ValueError, match="inside"):
            draw_point2d(img, torch.tensor([[-1, 0]], device=device), torch.tensor([1.0], device=device))
        assert img.sum() == 0  # nothing drawn, in particular not in the last column

    def test_draw_point2d_rounds_to_the_nearest_pixel_centre(self, dtype, device):
        img = torch.zeros(3, 4, dtype=dtype, device=device)
        draw_point2d(img, torch.tensor([[2.7, 1.6]], device=device), torch.tensor([1.0], device=device))
        assert img.nonzero().tolist() == [[2, 3]]  # (y, x): x 2.7 -> 3, y 1.6 -> 2

    @pytest.mark.parametrize(("x", "pixel"), [(-0.4, 0), (-0.5, 0), (0.5, 1), (1.5, 2), (2.5, 3)])
    def test_halfway_and_slightly_negative_coordinates(self, x, pixel, dtype, device):
        img = torch.zeros(3, 4, dtype=dtype, device=device)
        draw_point2d(img, torch.tensor([[x, 0.0]], device=device), torch.tensor([1.0], device=device))
        assert img[0].nonzero().flatten().tolist() == [pixel]

    @pytest.mark.parametrize("x", [129, 255])
    def test_half_precision_points_round_in_float32(self, x, device):
        # bfloat16 cannot represent x + 0.5 for x >= 128: 129 + 0.5 -> 130 and 255 + 0.5 -> 256 (a spurious ValueError)
        img = torch.zeros(1, 3, 256, device=device)
        points = torch.tensor([[x, 0]], dtype=torch.bfloat16, device=device)
        draw_point2d(img, points, torch.tensor([1.0], device=device))
        assert img[0, 0].nonzero().flatten().tolist() == [x]

    @pytest.mark.parametrize("point", [(-0.6, 0.0), (3.5, 0.0), (0.0, -0.6), (0.0, 2.5)])
    def test_draw_point2d_just_outside_after_rounding_raises(self, point, dtype, device):
        img = torch.zeros(3, 4, dtype=dtype, device=device)  # H = 3, W = 4: x 3.5 and y 2.5 round to just outside
        with pytest.raises(ValueError, match="inside"):
            draw_point2d(img, torch.tensor([point], device=device), torch.tensor([1.0], device=device))

    def test_float64_points_on_the_cpu_draw_on_the_image_device(self, device):
        # torch.from_numpy gives float64 points on the CPU; MPS has no float64, so they are rounded before they move
        img = torch.zeros(3, 4, 5, device=device)
        points = torch.tensor([[1.4, 2.6], [3.0, 0.0]], dtype=torch.float64)
        draw_point2d(img, points, torch.tensor([1.0, 0.5, 0.25]))
        assert img[0].nonzero().tolist() == [[0, 3], [3, 1]]  # (row, col): x 1.4 -> 1, y 2.6 -> 3
        assert img[:, 3, 1].tolist() == [1.0, 0.5, 0.25]

    def test_jit_script_matches_eager(self, device, dtype):
        img = torch.zeros(3, 4, 5, device=device, dtype=dtype)
        points = torch.tensor([[1, 2], [3, 0]], device=device)
        color = torch.tensor([1.0, 0.5, 0.25], device=device, dtype=dtype)
        scripted = torch.jit.script(draw_point2d)
        expected = draw_point2d(img.clone(), points, color)
        assert expected.abs().sum() > 0
        self.assert_close(scripted(img.clone(), points, color), expected, rtol=0, atol=0)


class TestDrawLine(BaseTester):
    def test_draw_line_vertical(self, dtype, device):
        """Test drawing a vertical line."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([6, 2]), torch.tensor([6, 0]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_line_horizontal(self, dtype, device):
        """Test drawing a horizontal line."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([6, 4]), torch.tensor([0, 4]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_line_with_big_coordinates(self, dtype, device):
        """Test drawing a line with big coordinates."""
        img = torch.zeros(1, 500, 500, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([200, 200]), torch.tensor([400, 200]), torch.tensor([255]))
        img_mask = torch.zeros(1, 500, 500, dtype=dtype, device=device)
        img_mask[:, 200, 200:401] = 255
        self.assert_close(img, img_mask)

    def test_draw_line_m_lte_neg1(self, dtype, device):
        """Test drawing a line with m <= -1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([0, 7]), torch.tensor([6, 0]), torch.tensor([255]))
        # Literal = 255 * cv2.line(np.zeros((8, 8), np.uint8), (0, 7), (6, 0), 1, 1, cv2.LINE_8) (OpenCV 5.0).
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_line_m_lt_0_gte_neg1(self, dtype, device):
        """Test drawing a line with -1 < m < 0."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([1, 5]), torch.tensor([7, 0]), torch.tensor([255]))
        # Literal = 255 * cv2.line(np.zeros((8, 8), np.uint8), (1, 5), (7, 0), 1, 1, cv2.LINE_8) (OpenCV 5.0).
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_line_m_gt_0_lt_1(self, dtype, device):
        """Test drawing a line with 0 < m < 1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([0, 0]), torch.tensor([6, 2]), torch.tensor([255]))
        # Literal = 255 * cv2.line(np.zeros((8, 8), np.uint8), (0, 0), (6, 2), 1, 1, cv2.LINE_8) (OpenCV 5.0).
        img_mask = torch.tensor(
            [
                [
                    [255.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_line_m_gte_1(self, dtype, device):
        """Test drawing a line with m >= 1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([3, 7]), torch.tensor([1, 4]), torch.tensor([255]))
        # Literal = 255 * cv2.line(np.zeros((8, 8), np.uint8), (3, 7), (1, 4), 1, 1, cv2.LINE_8) (OpenCV 5.0).
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    def test_draw_lines_batched(self, dtype, device):
        """Test drawing a line with m <= -1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(
            img, torch.tensor([[0, 7], [0, 7], [0, 2]]), torch.tensor([[6, 0], [0, 0], [7, 7]]), torch.tensor([255])
        )
        # Lines 1 and 3 equal cv2.line(np.zeros((8, 8), np.uint8), p1, p2, 1, 1, cv2.LINE_8) * 255 (OpenCV 5.0) for
        # (0, 7)->(6, 0) and (0, 2)->(7, 7); line 2 is the vertical x = 0.
        img_mask = torch.tensor(
            [
                [
                    [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [255.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                    [255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0],
                    [255.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 0.0, 255.0, 0.0, 255.0, 0.0, 0.0, 0.0],
                    [255.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0],
                    [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(img, img_mask)

    @pytest.mark.parametrize(
        "p1", [torch.tensor([-1, 0]), torch.tensor([0, -1]), torch.tensor([8, 0]), torch.tensor([0, 8])]
    )
    def test_p1_out_of_bounds(self, p1, dtype, device):
        """Tests that an exception is raised if p1 is out of bounds."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, p1, torch.tensor([0, 0]), torch.tensor([255]))

        assert "p1 is out of bounds." == str(excinfo.value)

    @pytest.mark.parametrize(
        "p2", [torch.tensor([-1, 0]), torch.tensor([0, -1]), torch.tensor([8, 0]), torch.tensor([0, 8])]
    )
    def test_p2_out_of_bounds(self, p2, dtype, device):
        """Tests that an exception is raised if p2 is out of bounds."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), p2, torch.tensor([255]))

        assert "p2 is out of bounds." == str(excinfo.value)

    @pytest.mark.parametrize(
        "p1,p2", [(torch.tensor([0, 0]), torch.tensor([0, 0])), (torch.tensor([0, 0]), torch.tensor([7, 7]))]
    )
    def test_p1_p2_bounds(self, p1, p2, dtype, device):
        """Tests to verify that points on the bounds do not trigger errors."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, p1, p2, torch.tensor([255]))
        assert img.shape == (1, 8, 8)

    @pytest.mark.parametrize("img_size", [(200, 100), (32, 3, 20, 20)])
    def test_image_size(self, img_size, dtype, device):
        img = torch.zeros(*img_size, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([255]))

        assert "image must have 3 dimensions (C,H,W)." == str(excinfo.value)

    @pytest.mark.parametrize(
        "img_size,color", [((1, 8, 8), torch.tensor([23, 53])), ((3, 8, 8), torch.tensor([23, 53]))]
    )
    def test_color_image_channel_size(self, img_size, color, dtype, device):
        img = torch.zeros(*img_size, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), torch.tensor([1, 1]), color)

        assert "color must have the same number of channels as the image." == str(excinfo.value)

    @pytest.mark.parametrize("p1,p2", [(torch.rand([10, 2]), torch.rand([20, 2])), (torch.rand([2]), torch.rand([3]))])
    def test_point_size(self, p1, p2, dtype, device):
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, p1, p2, torch.tensor([255]))

        assert "Input points must be 2D points with shape (2, ) or (B, 2) and must have the same batch sizes." == str(
            excinfo.value
        )

    # The tests below sit after test_point_size on purpose: its parametrize arguments call torch.rand at
    # import time, and testing/half_precision_eager_rng.py audits that call by line number (305).
    # Moving them above it shifts that line and fails the audit.
    def test_draw_line_lights_both_endpoints(self, dtype, device):
        # The integer ceiling reaches minor exactly at t = major, so p1 and p2 are both drawn. The float
        # step this replaced left p2 dark on about 7% of random lines.
        generator = torch.Generator().manual_seed(0)
        height, width = 37, 91
        xs = torch.randint(0, width, (300, 2), generator=generator)
        ys = torch.randint(0, height, (300, 2), generator=generator)
        color = torch.tensor([1.0])
        for (x1, x2), (y1, y2) in zip(xs.tolist(), ys.tolist()):
            img = torch.zeros(1, height, width, dtype=dtype, device=device)
            img = draw_line(img, torch.tensor([x1, y1]), torch.tensor([x2, y2]), color)
            assert img[0, y1, x1] == 1.0, ((x1, y1), (x2, y2))
            assert img[0, y2, x2] == 1.0, ((x1, y1), (x2, y2))
            assert img.sum() == max(abs(x2 - x1), abs(y2 - y1)) + 1, ((x1, y1), (x2, y2))

    def test_draw_line_empty_batch_draws_nothing(self, dtype, device):
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        empty = torch.zeros(0, 2, dtype=torch.long)
        out = draw_line(img, empty, empty, torch.tensor([1.0]))
        assert out.shape == (1, 8, 8)
        assert out.count_nonzero() == 0

    @pytest.mark.parametrize("p1, p2", [([1, 4], [6, 4]), ([[1, 1], [0, 7]], [[6, 6], [7, 0]])])
    def test_draw_line_draws_in_place_and_returns_the_input(self, p1, p2, dtype, device):
        """The Return section documents in-place drawing into, and returning, the input tensor."""
        img = torch.zeros(2, 8, 8, dtype=dtype, device=device)
        out = draw_line(img, torch.tensor(p1), torch.tensor(p2), torch.ones(2, dtype=dtype, device=device))
        assert out is img
        assert img.count_nonzero() > 0

    def test_draw_line_rejects_points_of_different_shapes(self, dtype, device):
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError, match="must have the same batch sizes"):
            draw_line(img, torch.tensor([1, 1]), torch.tensor([[2, 2], [3, 3]]), torch.tensor([1.0]))

    def test_draw_line_passes_through_its_exact_lattice_points(self, dtype, device):
        # The segment from (0, 0) to (117, 18) passes exactly through (13k, 2k). The minor coordinate
        # is ceil(t * 18 / 117), and a float step landed just above the exact integer at some of
        # these points, which rounded the pixel up a row: (26, 5) instead of (26, 4).
        img = torch.zeros(1, 19, 118, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([0, 0]), torch.tensor([117, 18]), torch.tensor([1.0]))
        for k in range(10):
            assert img[0, 2 * k, 13 * k] == 1.0, (13 * k, 2 * k)
            assert img[0, :, 13 * k].sum() == 1.0, 13 * k

    def test_draw_lines_batched_matches_drawing_each_line(self, dtype, device):
        # One batch mixing a vertical, a horizontal, a single point and both slope regimes.
        p1 = torch.tensor([[3, 1], [0, 9], [5, 5], [1, 2], [30, 0], [7, 18]])
        p2 = torch.tensor([[3, 17], [31, 9], [5, 5], [29, 11], [2, 19], [26, 3]])
        color = torch.tensor([1.0, 2.0])
        batched = draw_line(torch.zeros(2, 20, 32, dtype=dtype, device=device), p1, p2, color)
        expected = torch.zeros(2, 20, 32, dtype=dtype, device=device)
        for a, b in zip(p1, p2):
            expected = draw_line(expected, a, b, color)
        self.assert_close(batched, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("color", [torch.tensor([7.0]), torch.tensor(7.0)])
    def test_draw_line_single_value_color_fills_every_channel(self, color, dtype, device):
        img = torch.zeros(3, 4, 5, dtype=dtype, device=device)
        out = draw_line(img, torch.tensor([0, 1], device=device), torch.tensor([3, 1], device=device), color.to(device))
        assert out[:, 1, :4].tolist() == [[7.0] * 4] * 3
        assert int(out.count_nonzero()) == 12

    def test_draw_line_is_the_same_in_both_directions(self, dtype, device):
        p, q = torch.tensor([0, 0], device=device), torch.tensor([4, 1], device=device)
        a = draw_line(torch.zeros(1, 3, 6, dtype=dtype, device=device), p, q, torch.tensor([1.0], device=device))
        b = draw_line(torch.zeros(1, 3, 6, dtype=dtype, device=device), q, p, torch.tensor([1.0], device=device))
        self.assert_close(a, b)

    def test_draw_line_pixels_are_nearest_to_the_segment(self, dtype, device):
        # (0, 0) -> (4, 1) on 3x6: the ideal row is x / 4, so columns 0, 1 -> row 0, columns 3, 4 -> row 1 and the
        # tie at column 2 (y = 0.5) rounds up to row 1. OpenCV 5.0 rounds that tie down:
        # cv2.line(np.zeros((3, 6), np.uint8), (0, 0), (4, 1), 1, 1, cv2.LINE_8) returns
        # [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]].
        img = draw_line(
            torch.zeros(1, 3, 6, dtype=dtype, device=device),
            torch.tensor([0, 0], device=device),
            torch.tensor([4, 1], device=device),
            torch.tensor([1.0], device=device),
        )
        expected = torch.tensor(
            [[[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]]], dtype=dtype, device=device
        )
        self.assert_close(img, expected, rtol=0.0, atol=0.0)

    def test_draw_line_accepts_0d_scalar_color(self, dtype, device):
        # torch.tensor(255) is a natural grayscale color; color.size(0) used to IndexError.
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        out = draw_line(img, torch.tensor([1, 4]), torch.tensor([6, 4]), torch.tensor(255, dtype=dtype, device=device))
        assert out[0, 4, 1:7].tolist() == [255.0] * 6

    def test_draw_line_rounds_fractional_endpoints(self, dtype, device):
        img = torch.zeros(1, 3, 6, dtype=dtype, device=device)
        draw_line(
            img,
            torch.tensor([0.6, 1.0], device=device),
            torch.tensor([3.4, 1.0], device=device),
            torch.tensor([1.0], device=device),
        )
        assert img[0, 1].tolist() == [0, 1, 1, 1, 0, 0]

    def test_half_precision_endpoints_round_in_float32(self, device):
        # float16 spacing is 1 above 1024, so 1027 + 0.5 would round to 1028 and 1025 + 0.5 to 1026
        img = torch.zeros(1, 3, 1030, device=device)
        p1 = torch.tensor([1025.0, 1.0], dtype=torch.float16, device=device)
        p2 = torch.tensor([1027.0, 1.0], dtype=torch.float16, device=device)
        draw_line(img, p1, p2, torch.tensor([1.0], device=device))
        assert img[0, 1].nonzero().flatten().tolist() == [1025, 1026, 1027]

    def test_float64_endpoints_on_the_cpu_draw_on_the_image_device(self, device):
        # torch.from_numpy gives float64 points on the CPU; MPS has no float64, so they are rounded before they move
        img = torch.zeros(1, 4, 6, device=device)
        p1 = torch.tensor([0.4, 2.6], dtype=torch.float64)
        p2 = torch.tensor([4.6, 2.6], dtype=torch.float64)
        draw_line(img, p1, p2, torch.tensor([1.0]))
        assert img[0].nonzero().tolist() == [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5]]

    def test_draw_line_endpoint_rounding_outside_raises(self, dtype, device):
        img = torch.zeros(1, 3, 6, dtype=dtype, device=device)
        with pytest.raises(ValueError, match="p2 is out of bounds"):
            draw_line(
                img,
                torch.tensor([0.0, 1.0], device=device),
                torch.tensor([5.5, 1.0], device=device),
                torch.tensor([1.0], device=device),
            )


class TestDrawRectangle(BaseTester):
    def test_rejects_an_image_without_a_batch_axis(self, device):
        with pytest.raises(ShapeError):
            draw_rectangle(torch.zeros(3, 6, 5, device=device), torch.tensor([[[0, 0, 2, 2]]], device=device))

    @pytest.mark.parametrize("batch", [4, 17])
    @pytest.mark.parametrize("color", [torch.Tensor([1.0]), torch.Tensor([0.5])])
    def test_smoke(self, device, batch, color):
        black_image = torch.zeros(batch, 1, 3, 3, device=device)  # 1 channel 3x3 black_image
        points = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device).expand(batch, 1, 4)  # single pixel rectangle

        draw_rectangle(black_image, points, color=color)

        target = torch.zeros(batch, 1, 3, 3, device=device)
        target[:, :, 1, 1] = color

        assert torch.all(black_image == target)

    @pytest.mark.parametrize("batch", [8, 11])
    @pytest.mark.parametrize("fill", [True, False])
    @pytest.mark.parametrize("height", [12, 106, 298])
    @pytest.mark.parametrize("width", [7, 123, 537])
    def test_fill_and_edges(self, device, batch, fill, height, width):
        black_image = torch.zeros(batch, 3, height, width, device=device)
        # we should pass height - 1 and width - 1 but rectangle should clip correctly
        points = torch.tensor([0, 0, width, height]).to(device).expand(batch, 1, 4)

        image_w_rectangle = draw_rectangle(black_image, points, color=torch.tensor([1.0]), fill=fill)

        assert image_w_rectangle is black_image
        if fill:
            assert image_w_rectangle.sum() == batch * 3 * height * width
        else:
            # the right side (x = width) and bottom side (y = height) lie outside the image and are not drawn
            assert image_w_rectangle.sum() == batch * 3 * (height + width - 1)

    @pytest.mark.parametrize("batch", [4, 6])
    @pytest.mark.parametrize("N", [5, 12])
    @pytest.mark.parametrize("fill", [True, False])
    def test_n_rectangles(self, device, batch, N, fill):
        points_list = []
        h, w = 20, 20
        for b in range(batch):
            points_list.append([])
            for n in range(N):
                points_list[b].append([])
                points_list[b][n].append(int(torch.randint(0, w - 1, (1,))))
                points_list[b][n].append(int(torch.randint(0, h - 1, (1,))))
                points_list[b][n].append(int(torch.randint(points_list[b][n][-2] + 1, w, (1,))))
                points_list[b][n].append(int(torch.randint(points_list[b][n][-2] + 1, h, (1,))))

        points = torch.tensor(points_list, device=device)

        random_background = torch.rand(batch, 3, h, w, device=device)
        random_w_rectangle = random_background.clone()

        draw_rectangle(random_w_rectangle, points, color=torch.tensor([1.0, 1.0, 1.0]), fill=fill)

        for b in range(batch):
            for n in range(N):
                if fill:
                    assert (
                        random_w_rectangle[
                            b,
                            :,
                            points_list[b][n][1] : points_list[b][n][3] + 1,
                            points_list[b][n][0] : points_list[b][n][2] + 1,
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1)
                        * (points_list[b][n][2] - points_list[b][n][0] + 1)
                        * 3
                    )
                else:
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1] : points_list[b][n][3] + 1, points_list[b][n][0]
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1] : points_list[b][n][3] + 1, points_list[b][n][2]
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1], points_list[b][n][0] : points_list[b][n][2] + 1
                        ].sum()
                        == (points_list[b][n][2] - points_list[b][n][0] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1], points_list[b][n][0] : points_list[b][n][2] + 1
                        ].sum()
                        == (points_list[b][n][2] - points_list[b][n][0] + 1) * 3
                    )

    @pytest.mark.parametrize("color", [torch.tensor([0.5, 0.3, 0.15]), torch.tensor([0.23, 0.33, 0.8])])
    def test_color_background(self, device, color):
        image = torch.zeros(1, 3, 40, 40, device=device)
        image[:, 0, :, :] = color[0]
        image[:, 1, :, :] = color[1]
        image[:, 2, :, :] = color[2]
        image_w_rectangle = image.clone()
        p1 = (1, 5)
        p2 = (30, 39)
        points = torch.tensor([[[p1[1], p1[0], p2[1], p2[0]]]], device=device)

        draw_rectangle(image_w_rectangle, points, color=torch.tensor([1.0]))
        assert (
            torch.abs(
                (image_w_rectangle - image).sum()
                - (1 - color[0]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
                - (1 - color[1]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
                - (1 - color[2]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
            )
            <= 0.0001
        )

    @pytest.mark.parametrize("color", [torch.tensor([0.34, 0.63, 0.16]), torch.tensor([0.29, 0.13, 0.48])])
    def test_color_foreground(self, device, color):
        image = torch.zeros(1, 3, 50, 40, device=device)
        image_w_rectangle = image.clone()
        p1 = (10, 4)
        p2 = (11, 40)
        points = torch.tensor([[[p1[1], p1[0], p2[1], p2[0]]]], device=device)

        draw_rectangle(image_w_rectangle, points, color=color)

        # corners are double counted, no plus 1 for y since p2[1] of 40 already lies outside of the image
        assert (
            torch.abs(
                (image_w_rectangle - image).sum()
                - (color[0]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
                - (color[1]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
                - (color[2]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
            )
            <= 0.0001
        )

    @pytest.mark.parametrize("fill", [False, True])
    def test_0d_scalar_color_matches_one_channel_color(self, fill, dtype, device):
        """A 0-d color used to fail unpacking (b, n, c); it now draws like a size-1 color."""
        rect = torch.tensor([[[1, 1, 3, 3]]], device=device)
        out = draw_rectangle(
            torch.zeros(1, 1, 5, 5, dtype=dtype, device=device),
            rect,
            torch.tensor(9.0, dtype=dtype, device=device),
            fill=fill,
        )
        expected = draw_rectangle(
            torch.zeros(1, 1, 5, 5, dtype=dtype, device=device),
            rect,
            torch.tensor([9.0], dtype=dtype, device=device),
            fill=fill,
        )
        self.assert_close(out, expected)
        assert out.count_nonzero() > 0

    def test_reversed_corners_draw_the_same_rectangle(self, device):
        a = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device), torch.tensor([[[1, 1, 3, 2]]], device=device), torch.tensor([1.0])
        )
        b = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device), torch.tensor([[[3, 2, 1, 1]]], device=device), torch.tensor([1.0])
        )
        assert a.sum() > 0
        self.assert_close(a, b)

    def test_sides_outside_the_image_are_not_drawn(self, device):
        out = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device),
            torch.tensor([[[-2, -2, 10, 10]]], device=device),
            torch.tensor([1.0]),
        )
        assert out.sum() == 0  # every side lies outside the 5x6 image

    def test_partly_outside_box_draws_only_its_inside_sides(self, device):
        out = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device), torch.tensor([[[2, 1, 9, 3]]], device=device), torch.tensor([1.0])
        )
        expected = torch.zeros(5, 6, device=device)
        expected[1, 2:] = 1  # top side, clipped at the right border
        expected[3, 2:] = 1  # bottom side
        expected[1:4, 2] = 1  # left side; the right side (x = 9) is outside and not drawn
        self.assert_close(out[0, 0], expected)

    def test_fill_is_clipped_to_the_image(self, device):
        out = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device),
            torch.tensor([[[-2, 3, 2, 9]]], device=device),
            torch.tensor([1.0]),
            fill=True,
        )
        expected = torch.zeros(5, 6, device=device)
        expected[3:, :3] = 1
        self.assert_close(out[0, 0], expected)

    def test_rectangle_rounds_fractional_corners(self, device):
        out = draw_rectangle(
            torch.zeros(1, 1, 5, 6, device=device),
            torch.tensor([[[0.6, 0.6, 2.4, 2.6]]], device=device),
            torch.tensor([1.0]),
            fill=True,
        )
        expected = torch.zeros(5, 6, device=device)
        expected[1:4, 1:3] = 1  # x 0.6..2.4 -> 1..2, y 0.6..2.6 -> 1..3
        self.assert_close(out[0, 0], expected)

    @pytest.mark.parametrize(("rectangle", "match"), [(torch.zeros(2, 1, 4), "batch"), (torch.zeros(1, 1, 3), "4")])
    def test_invalid_rectangle_raises_value_error(self, rectangle, match, device):
        with pytest.raises(ValueError, match=match):
            draw_rectangle(torch.zeros(1, 1, 5, 6, device=device), rectangle.to(device))

    def test_full_frame_inside_the_image(self, device):
        batch, height, width = 2, 7, 9
        black_image = torch.zeros(batch, 3, height, width, device=device)
        points = torch.tensor([0, 0, width - 1, height - 1]).to(device).expand(batch, 1, 4)
        out = draw_rectangle(black_image, points, color=torch.tensor([1.0]))
        assert out.sum() == batch * 3 * (2 * height + 2 * width - 4)

    @pytest.mark.parametrize("fill", [False, True])
    def test_jit_script_matches_eager(self, device, dtype, fill):
        # draw_rectangle compiles under torch.jit.script and matches eager; H != W, the second box has reversed corners
        image = torch.zeros(2, 3, 7, 10, device=device, dtype=dtype)
        rectangle = torch.tensor(
            [[[1.0, 1.0, 6.0, 3.0]], [[8.0, 5.0, 2.0, 0.0]]], device=device, dtype=dtype
        )  # reversed corners
        color = torch.tensor([[[1.0, 0.5, 0.25]], [[0.25, 0.5, 1.0]]], device=device, dtype=dtype)
        scripted = torch.jit.script(draw_rectangle)
        expected = draw_rectangle(image.clone(), rectangle, color, fill=fill)
        assert expected.abs().sum() > 0
        self.assert_close(scripted(image.clone(), rectangle, color, fill=fill), expected, rtol=0, atol=0)

    @pytest.mark.device_agnostic
    @pytest.mark.skipif(not dynamo_is_available(), reason="torch.export needs Dynamo on this torch/python pair")
    @pytest.mark.skipif(
        not torch_version_ge(2, 14),
        reason="slices with data-dependent bounds export from torch 2.14 (2.9.1 fails to export, 2.6 bakes the input)",
    )
    @pytest.mark.parametrize("fill", [False, True])
    def test_onnx_export_draws_the_rectangles_given_at_run_time(self, fill):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        ort = pytest.importorskip("onnxruntime")

        class Draw(torch.nn.Module):
            def forward(self, image, rectangle, color):
                return draw_rectangle(image.clone(), rectangle, color, fill=fill)

        model = Draw()
        image = torch.zeros(1, 3, 6, 9)
        color = torch.tensor([1.0, 0.5, 0.2])
        traced_on = torch.tensor([[[1.0, 1.0, 4.0, 3.0], [2.0, 0.0, 7.0, 5.0]]])
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (image, traced_on, color), dynamo=True, opset_version=18, verbose=False).save(
                buffer
            )
        session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        assert len(names) == 3  # the rectangle stays a graph input, not a constant
        # rectangles other than the traced ones: reversed and fractional corners, sides and whole boxes outside
        for rectangle in (
            traced_on,
            torch.tensor([[[7.0, 4.0, 2.0, 1.0], [-3.0, 2.4, 3.6, 12.0]]]),
            torch.tensor([[[20.0, 20.0, 30.0, 30.0], [-5.0, -5.0, 0.4, 0.6]]]),
            torch.tensor([[[0.0, 0.0, 8.0, 5.0], [8.0, 5.0, 8.0, 5.0]]]),
        ):
            feeds = dict(zip(names, (image.numpy(), rectangle.numpy(), color.numpy())))
            got = torch.from_numpy(session.run(None, feeds)[0])
            self.assert_close(got, model(image, rectangle, color), rtol=0, atol=0)


class TestFillConvexPolygon(BaseTester):
    @pytest.mark.parametrize("vertex_dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_edge_maths_does_not_depend_on_the_vertex_dtype(self, vertex_dtype, device, dtype):
        # vertices 260 and 280 are exact in bfloat16; the fill must be columns 260..280 (21 px) in every row
        poly = torch.tensor([[[260.0, 0.0], [280.0, 0.0], [280.0, 5.0], [260.0, 5.0]]], device=device).to(vertex_dtype)
        img = draw_convex_polygon(
            torch.zeros(1, 1, 6, 300, device=device, dtype=dtype), poly, torch.tensor([1.0], device=device, dtype=dtype)
        )
        assert img[0, 0, 2].nonzero().flatten()[[0, -1]].tolist() == [260, 280]
        assert int(img[0, 0, 2].sum()) == 21

    def test_centre_exactly_on_a_sloped_edge_is_filled(self, device, dtype):
        # hull [[13,18],[48,1],[12,31],[12,22]] on 31x47: (row 26, col 18) lies on edge (48,1)->(12,31)
        poly = torch.tensor([[[13.0, 18.0], [48.0, 1.0], [12.0, 31.0], [12.0, 22.0]]], device=device)
        img = draw_convex_polygon(torch.zeros(1, 1, 31, 47, device=device), poly, torch.tensor([1.0], device=device))
        assert img[0, 0, 26, 18] == 1

    def test_long_shallow_edge_with_vertices_outside_the_image(self, device, dtype):
        # triangle (-100,0),(100,4),(-100,4) on 5x6 covers 13 pixel centres (rows 2-4)
        poly = torch.tensor([[[-100.0, 0.0], [100.0, 4.0], [-100.0, 4.0]]], device=device)
        img = draw_convex_polygon(torch.zeros(1, 1, 5, 6, device=device), poly, torch.tensor([1.0], device=device))
        assert int(img.sum()) == 13

    def test_circle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        n = 5000
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)[None].expand(b, n)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        x = (2 * math.pi * t).cos()
        y = (2 * math.pi * t).sin()
        ctr = 200
        radius = 200
        pts = ctr + radius * torch.stack((x, y), dim=-1)
        poly_im = draw_convex_polygon(im, pts, color)
        XY = create_meshgrid(h, w, normalized_coordinates=False, device=device, dtype=dtype)
        inside = (((XY[..., 1] - ctr) ** 2 + (XY[..., 0] - ctr) ** 2).sqrt() <= radius)[:, None].expand(b, c, h, w)
        circ_im = inside * color[..., None, None]
        assert (circ_im - poly_im).abs().mean() <= 1e-4

    def test_ellipse(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        n = 5000
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)[None].expand(b, n)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        lam = 2
        x = lam * (2 * math.pi * t).cos()
        y = (2 * math.pi * t).sin()
        ctr = 200
        radius = 100
        pts = ctr + radius * torch.stack((x, y), dim=-1)
        poly_im = draw_convex_polygon(im, pts, color)
        XY = create_meshgrid(h, w, normalized_coordinates=False, device=device, dtype=dtype)
        inside = (((XY[..., 1] - ctr) ** 2 + ((XY[..., 0] - ctr) / lam) ** 2).sqrt() <= radius)[:, None].expand(
            b, c, h, w
        )
        ellipse_im = inside * color[..., None, None]
        assert (ellipse_im - poly_im).abs().mean() <= 1e-4

    def test_rectangle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        pts = torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        self.assert_close(rect_im, poly_im)

    def test_batch(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = torch.tensor(
            [[[4, 4], [12, 4], [12, 8], [4, 8]], [[0, 0], [4, 0], [4, 4], [0, 4]]], dtype=dtype, device=device
        )
        color = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        self.assert_close(rect_im, poly_im)

    def test_batch_variable_size(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = [
            torch.tensor([[4, 4], [12, 4], [12, 8], [4, 8]], dtype=dtype, device=device),
            torch.tensor([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]], dtype=dtype, device=device),
        ]
        color = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        self.assert_close(rect_im, poly_im)

    def test_batch_color_no_batch(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = [
            torch.tensor([[4, 4], [12, 4], [12, 8], [4, 8]], dtype=dtype, device=device),
            torch.tensor([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]], dtype=dtype, device=device),
        ]
        color = torch.tensor([0.5, 0.5, 0.75], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color, fill=True)
        self.assert_close(rect_im, poly_im)

    def test_out_of_bounds_rectangle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        pts = 350 + torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        self.assert_close(rect_im, poly_im)
        pts = -150 + torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        self.assert_close(rect_im, poly_im)

    def test_empty_polygon_leaves_image_unchanged(self, device, dtype):
        """An empty (B, 0, 2) polygon batch must not IndexError on loop close."""
        im = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)
        pts = torch.zeros(1, 0, 2, device=device, dtype=dtype)
        color = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype)
        out = draw_convex_polygon(im.clone(), pts, color)
        self.assert_close(out, im)

    def test_empty_polygon_is_still_validated(self, device, dtype):
        im = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)
        color = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype)
        with pytest.raises(BaseError, match="same batch dimension"):
            draw_convex_polygon(im, torch.zeros(2, 0, 2, device=device, dtype=dtype), color)
        with pytest.raises(BaseError, match="xy"):
            draw_convex_polygon(im, torch.zeros(1, 0, 3, device=device, dtype=dtype), color)

    def test_single_vertex_fills_its_pixel(self, device, dtype):
        """A one-vertex polygon draws that point, like a zero-length two-vertex polygon."""
        im = torch.rand(1, 3, 12, 16, device=device, dtype=dtype)
        color = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=dtype)
        vertex = torch.tensor([[[4.0, 4.0]]], device=device, dtype=dtype)
        out = draw_convex_polygon(im.clone(), vertex, color)
        expected = draw_convex_polygon(im.clone(), vertex.expand(1, 2, 2), color)
        self.assert_close(out, expected)
        self.assert_close(out[..., 4, 4], color)

    def test_empty_polygon_in_a_list_leaves_its_image_unchanged(self, device, dtype):
        """An empty polygon has no vertex to pad with; the rest of the batch is still drawn."""
        im = torch.rand(2, 3, 12, 16, device=device, dtype=dtype)
        square = torch.tensor([[4, 4], [12, 4], [12, 8], [4, 8]], device=device, dtype=dtype)
        color = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], device=device, dtype=dtype)
        out = draw_convex_polygon(im.clone(), [torch.zeros(0, 2, device=device, dtype=dtype), square], color)
        self.assert_close(out[:1], im[:1])
        self.assert_close(out[1:], draw_convex_polygon(im[1:].clone(), square[None], color[1:]))

    def test_empty_list_with_empty_batch(self, device, dtype):
        im = torch.rand(0, 3, 12, 16, device=device, dtype=dtype)
        color = torch.zeros(0, 3, device=device, dtype=dtype)
        out = draw_convex_polygon(im.clone(), [], color)
        assert out.shape == im.shape

    def test_sloped_edge_fills_the_pixel_it_passes_through(self, device, dtype):
        """The right edge of this triangle crosses x = 4 + 2 * (y - 4) exactly, and that pixel is inside."""
        tri = torch.tensor([[[4.0, 4.0], [12.0, 8.0], [4.0, 8.0]]], device=device, dtype=dtype)
        im = torch.zeros(1, 1, 12, 16, device=device, dtype=dtype)
        out = draw_convex_polygon(im, tri, torch.ones(1, 1, device=device, dtype=dtype))
        ys = torch.arange(12, device=device)[:, None]
        xs = torch.arange(16, device=device)[None, :]
        expected = (ys >= 4) & (ys <= 8) & (xs >= 4) & (xs <= 4 + 2 * (ys - 4))
        assert out[0, 0].eq(expected).all()

    def test_0d_scalar_color_matches_one_channel_color(self, device, dtype):
        """A 0-d color used to fail unpacking (b, c); it now fills like a size-1 color."""
        square = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        im = torch.zeros(1, 1, 5, 5, device=device, dtype=dtype)
        out = draw_convex_polygon(im.clone(), square, torch.tensor(9.0, device=device, dtype=dtype))
        expected = draw_convex_polygon(im.clone(), square, torch.tensor([9.0], device=device, dtype=dtype))
        self.assert_close(out, expected)
        assert out.count_nonzero() > 0

    def test_uint8_image_takes_a_float_colour(self, device):
        img = torch.full((1, 3, 5, 7), 9, dtype=torch.uint8, device=device)
        poly = torch.tensor([[[1.0, 1.0], [5.0, 1.0], [5.0, 3.0], [1.0, 3.0]]], device=device)
        draw_convex_polygon(img, poly, torch.tensor([[200.0, 100.0, 50.0]], device=device))
        assert img.dtype == torch.uint8
        assert img[0, :, 2, 3].tolist() == [200, 100, 50]  # inside: the colour, cast to uint8
        assert img[0, :, 4, 6].tolist() == [9, 9, 9]  # outside: unchanged
        assert int((img[0, 0] == 200).sum()) == 15  # rows 1-3, columns 1-5

    def test_nan_pixel_inside_the_polygon_is_overwritten(self, device, dtype):
        img = torch.zeros(1, 1, 5, 6, device=device, dtype=dtype)
        img[0, 0, 2, 2] = float("nan")
        poly = torch.tensor([[[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        draw_convex_polygon(img, poly, torch.tensor([1.0], device=device, dtype=dtype))
        assert img[0, 0, 2, 2].item() == 1.0
        assert not img.isnan().any()


class TestConventionsDraw(BaseTester):
    """Pins for the (x, y) order, inclusive corners, polygon edge rule and in-place drawing."""

    def test_convention_draw_coordinates_are_x_then_y(self, device, dtype):
        img = torch.zeros(1, 3, 5, device=device, dtype=dtype)  # H = 3, W = 5
        draw_point2d(img, torch.tensor([[4, 1]], device=device), torch.tensor([1.0], device=device))
        assert img[0].nonzero().tolist() == [[1, 4]]  # (row y, column x)
        transposed = torch.zeros(1, 5, 3, device=device, dtype=dtype)
        draw_point2d(transposed, torch.tensor([[1, 4]], device=device), torch.tensor([1.0], device=device))
        self.assert_close(transposed[0], img[0].T)

    def test_convention_rectangle_includes_both_corners(self, device, dtype):
        img = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)  # H = 5, W = 7
        rect = torch.tensor([[[1, 0, 4, 2]]], device=device)  # x1, y1, x2, y2
        out = draw_rectangle(img, rect, torch.tensor([1.0], device=device), fill=True)
        assert out[0, 0].nonzero().min(0).values.tolist() == [0, 1]
        assert out[0, 0].nonzero().max(0).values.tolist() == [2, 4]  # (row y2, column x2) is drawn
        assert out.sum().item() == 3 * 4  # rows 0..2 x columns 1..4: (y2 - y1 + 1) * (x2 - x1 + 1)
        outline = draw_rectangle(torch.zeros_like(img), rect, torch.tensor([1.0], device=device))
        assert outline[0, 0, 2, 4].item() == 1.0
        assert outline[0, 0, 1, 2].item() == 0.0  # the interior of an outline stays untouched
        transposed = draw_rectangle(
            torch.zeros(1, 1, 7, 5, device=device, dtype=dtype),
            torch.tensor([[[0, 1, 2, 4]]], device=device),
            torch.tensor([1.0], device=device),
            fill=True,
        )
        self.assert_close(transposed[0, 0], out[0, 0].T)

    def test_convention_polygon_fills_centres_on_or_inside_its_edges(self, device, dtype):
        # A pixel is filled when its centre is on or inside the polygon: edges at x = 1.5 and 3.5 fill columns 2 and
        # 3 only; an edge exactly through x = 1 includes column 1.
        colors = torch.tensor([1.0], device=device, dtype=dtype)
        img = torch.zeros(1, 1, 4, 6, device=device, dtype=dtype)  # H = 4, W = 6
        poly = torch.tensor([[[1.5, 0.0], [3.5, 0.0], [3.5, 3.0], [1.5, 3.0]]], device=device, dtype=dtype)
        out = draw_convex_polygon(img, poly, colors)
        assert out[0, 0].nonzero()[:, 1].unique().tolist() == [2, 3]
        assert out[0, 0].nonzero()[:, 0].unique().tolist() == [0, 1, 2, 3]
        on_edge = torch.tensor([[[1.0, 0.0], [3.5, 0.0], [3.5, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        out = draw_convex_polygon(torch.zeros_like(img), on_edge, colors)
        assert out[0, 0].nonzero()[:, 1].unique().tolist() == [1, 2, 3]
        transposed = draw_convex_polygon(torch.zeros(1, 1, 6, 4, device=device, dtype=dtype), poly.flip(-1), colors)
        self.assert_close(transposed[0, 0], draw_convex_polygon(torch.zeros_like(img), poly, colors)[0, 0].T)

    def test_convention_concave_polygon_fills_each_scanline_span(self, device, dtype):
        # draw_convex_polygon fills, on each scanline (row), everything from the leftmost to the rightmost edge
        # crossing, so the notch of a concave "V" is filled. Rows 0..3 span columns 1..5. Outside the polygon are
        # only row 0 columns 2..4 and row 1 column 3 (columns 2 and 4 of row 1 lie on the notch edges, which count as
        # inside); they are filled anyway.
        color = torch.tensor([1.0], device=device, dtype=dtype)
        notched = torch.tensor(
            [[[1.0, 0.0], [3.0, 2.0], [5.0, 0.0], [5.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype
        )  # (x, y)
        out = draw_convex_polygon(torch.zeros(1, 1, 5, 7, device=device, dtype=dtype), notched, color)  # H = 5, W = 7
        expected = torch.zeros(5, 7, device=device, dtype=dtype)
        expected[0:4, 1:6] = 1.0
        self.assert_close(out[0, 0], expected)

        # The span is per row, not per column: the same polygon with x and y swapped (a ">" whose notch opens to the
        # left) is filled exactly as the polygon, with no notch filled in (the scanlines are still rows).
        # Snippet: rows y = 1..5 span [min x, max x] over the edges (0,1)-(2,3), (2,3)-(0,5), x = 3 and y = 1 / 5:
        #   y=1: 0..3, y=2: 1..3, y=3: 2..3, y=4: 1..3, y=5: 0..3
        swapped = draw_convex_polygon(
            torch.zeros(1, 1, 7, 5, device=device, dtype=dtype), notched.flip(-1), color
        )  # H = 7, W = 5
        expected_swapped = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            device=device,
            dtype=dtype,
        )
        self.assert_close(swapped[0, 0], expected_swapped)

    @pytest.mark.parametrize("name", ["draw_point2d", "draw_line", "draw_rectangle", "draw_convex_polygon"])
    def test_convention_draw_functions_draw_in_place(self, device, dtype, name):
        # Each function writes into the tensor it is given and returns that same object.
        color = torch.tensor([1.0], device=device, dtype=dtype)
        batch = torch.zeros(1, 1, 5, 7, device=device, dtype=dtype)  # H = 5, W = 7
        if name == "draw_point2d":
            target = batch[0]  # (C, H, W)
            out = draw_point2d(target, torch.tensor([[3, 1]], device=device), color)
        elif name == "draw_line":
            target = batch[0]  # (C, H, W)
            out = draw_line(target, torch.tensor([0, 0], device=device), torch.tensor([3, 2], device=device), color)
        elif name == "draw_rectangle":
            target = batch
            out = draw_rectangle(target, torch.tensor([[[1, 1, 3, 2]]], device=device), color, fill=True)
        else:
            target = batch
            poly = torch.tensor([[[1.0, 1.0], [3.0, 1.0], [3.0, 2.0]]], device=device, dtype=dtype)
            out = draw_convex_polygon(target, poly, color)
        assert out is target
        assert batch.sum().item() > 0  # the caller's storage was written
