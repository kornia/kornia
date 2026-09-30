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
import warnings
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from kornia.color import rgb_to_grayscale
from kornia.core._compat import torch_version, torch_version_ge
from kornia.core.exceptions import BaseError, ImageError
from kornia.filters import Canny, canny, gaussian_blur2d, sobel, spatial_gradient

from testing.base import BaseTester, supports_reflect_padding, supports_replicate_padding


class TestCanny(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("kernel_size", [3, (5, 7)])
    @pytest.mark.parametrize("sigma", [(1.5, 1.0), (2.5, 0.5)])
    @pytest.mark.parametrize("hysteresis", [False, True])
    @pytest.mark.parametrize("low_threshold,high_threshold", [(0.1, 0.2), (0.3, 0.5)])
    def test_smoke(self, batch_size, kernel_size, sigma, hysteresis, low_threshold, high_threshold, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)

        op = Canny(low_threshold, high_threshold, kernel_size, sigma, hysteresis)
        actual = op(inp)
        assert len(actual) == 2
        assert actual[0].shape == (batch_size, 1, 4, 4)
        assert actual[1].shape == (batch_size, 1, 4, 4)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cardinality(self, batch_size, device, dtype):
        inp = torch.zeros(batch_size, 3, 4, 4, device=device, dtype=dtype)

        op = Canny()
        magnitude, edges = op(inp)

        assert magnitude.shape == (batch_size, 1, 4, 4)
        assert edges.shape == (batch_size, 1, 4, 4)

    @pytest.mark.parametrize("as_module", [False, True])
    @pytest.mark.parametrize("with_weak_edges", [False, True])
    def test_hysteresis_preserves_dtype(self, device, dtype, as_module, with_weak_edges):
        inp = torch.tensor(
            [
                [0.5, 0.4, 0.5, 0.45, 0.1],
                [0.3, 0.2, 0.3, 0.0, 0.3],
                [0.5, 1.0, 1.0, 0.6, 0.75],
                [0.2, 0.4, 0.6, 0.0, 0.5],
                [0.1, 0.35, 0.35, 0.26, 0.1],
            ],
            device=device,
            dtype=dtype,
        ).view(1, 1, 5, 5)
        if not with_weak_edges:
            inp.zero_()
        _, thresholded = canny(inp, hysteresis=False)
        if with_weak_edges:
            # This weak edge is adjacent to a strong one and must survive hysteresis.
            assert thresholded[0, 0, 2, 0] == 0.5
            assert thresholded[0, 0, 1, 0] == 1.0

        magnitude, edges = Canny()(inp) if as_module else canny(inp)

        assert magnitude.dtype == edges.dtype == dtype
        assert magnitude.device == edges.device == inp.device
        assert magnitude.shape == edges.shape == inp.shape
        assert ((edges == 0) | (edges == 1)).all()
        if with_weak_edges:
            assert edges[0, 0, 2, 0] == 1.0
        else:
            assert (edges == 0).all()

    def test_exception(self, device, dtype):
        from kornia.core.exceptions import BaseError, ShapeError, TypeCheckError

        with pytest.raises(BaseError) as errinfo:
            Canny(0.3, 0.2)
        assert "low_threshold should be smaller than or equal to the high_threshold" in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            Canny(-2, 0.3)
        assert "Invalid low threshold." in str(errinfo.value)

        with pytest.raises(BaseError) as errinfo:
            Canny(0, 3)
        assert "Invalid low threshold." in str(errinfo.value)

        with pytest.raises(TypeCheckError) as errinfo:
            canny(1)
        assert "Type mismatch: expected Tensor" in str(errinfo.value)

        inp = torch.zeros(3, 4, 4, device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            canny(inp)
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_noncontiguous(self, batch_size, device, dtype):
        inp = torch.rand(batch_size, 3, 5, 5, device=device, dtype=dtype).expand(batch_size, -1, -1, -1)

        magnitude, edges = canny(inp)

        assert magnitude.is_contiguous()
        assert edges.is_contiguous()

    def test_magnitude(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.2458, 0.9672, 1.2458, 0.0],
                        [0.0, 0.9672, 0.0, 0.9672, 0.0],
                        [0.0, 1.2458, 0.9672, 1.2458, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = canny(inp)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

    def test_magnitude_hyst(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [1.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = canny(inp, hysteresis=True)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

    def test_magnitude_hyst_false(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.5000, 0.0000, 0.0000, 1.0000, 0.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = canny(inp, hysteresis=False)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

    def test_magnitude_threshold(self, device, dtype):
        inp = torch.tensor(
            [
                [
                    [
                        [0.5, 0.4, 0.5, 0.45, 0.1],
                        [0.3, 0.2, 0.3, 0.0, 0.3],
                        [0.5, 1.0, 1.0, 0.6, 0.75],
                        [0.2, 0.4, 0.6, 0.0, 0.5],
                        [0.1, 0.35, 0.35, 0.26, 0.1],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_magnitude = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.4858, 0.5594, 0.6878, 0.6977, 0.5602],
                        [0.1129, 0.0000, 0.0000, 0.4531, 0.0000],
                        [0.6115, 0.5859, 0.6110, 0.6766, 0.5160],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_edges = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        magnitude, edges = canny(inp, low_threshold=0.3, high_threshold=0.9)

        self.assert_close(magnitude, expected_magnitude, atol=1e-4, rtol=1e-4)
        self.assert_close(edges, expected_edges, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        if "cuda" in str(device):
            pytest.skip("RuntimeError: Backward is not reentrant, i.e., running backward,")
        batch_size, channels, height, width = 1, 1, 3, 4
        img = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)
        self.gradcheck(canny, img)

    def test_module(self, device, dtype):
        img = torch.rand(2, 3, 4, 5, device=device, dtype=dtype)
        op = canny
        op_module = Canny()
        expected_magnitude, expected_edges = op(img)
        actual_magnitude, actual_edges = op_module(img)
        self.assert_close(actual_magnitude, expected_magnitude)
        self.assert_close(actual_edges, expected_edges)

    @pytest.mark.parametrize("kernel_size", [5, (5, 7)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.skipif(torch_version() in {"2.0.0", "2.0.1"}, reason="Not working on 2.0")
    def test_dynamo(self, batch_size, kernel_size, device, dtype, torch_optimizer):
        if (
            torch_version() in {"2.1.1", "2.1.2", "2.2.2", "2.3.1"}
            and dtype == torch.float64
            and (isinstance(kernel_size, int) or kernel_size[0] == kernel_size[1])
        ):
            pytest.skip("Canny compiled failing into fp64 for kernel sizes where kx and ky are equals")
        data = torch.ones(batch_size, 3, 10, 10, device=device, dtype=dtype)
        op = Canny(kernel_size=kernel_size)
        op_optimized = torch_optimizer(op)

        expected_magnitude, expected_edges = op(data)
        actual_magnitude, actual_edges = op_optimized(data)

        self.assert_close(actual_magnitude, expected_magnitude)
        self.assert_close(actual_edges, expected_edges)

    @pytest.mark.parametrize("hysteresis", [False, True])
    @pytest.mark.parametrize("image", ["random", "rectangle"])
    def test_dynamo_on_edges(self, image, hysteresis, device, dtype, torch_optimizer):
        self._require_padding(device, dtype)
        if dtype in (torch.float16, torch.bfloat16) and image == "random":
            pytest.skip("the compiled graph computes in float32 and rounds once, so near-ties may resolve differently")
        # test_dynamo's constant image has no edge. A random image, blurred, has no exact tie; a rectangle with an inner
        # step, unblurred, has exact ties that compiled and eager arithmetic resolve alike, in every dtype. Both
        # exercise the suppression and the hysteresis in the compiled graph
        if image == "random":
            generator = torch.Generator().manual_seed(0)
            img = torch.rand(1, 3, 40, 48, generator=generator).to(device=device, dtype=dtype)
            op = Canny(hysteresis=hysteresis)
        else:
            img = torch.zeros(1, 3, 40, 48, device=device, dtype=dtype)
            img[..., 8:32, 10:38] = 1.0
            img[..., 16:24, 18:30] = 0.5
            op = Canny(kernel_size=1, hysteresis=hysteresis)
        expected_magnitude, expected_edges = op(img)
        actual_magnitude, actual_edges = torch_optimizer(op)(img)
        assert expected_edges.sum().item() > 0
        self.assert_close(actual_edges, expected_edges)
        self.assert_close(actual_magnitude, expected_magnitude)

    @staticmethod
    def _require_padding(device, dtype):
        # gaussian_blur2d pads by reflection and spatial_gradient by replication, even with kernel_size=1
        if not (supports_reflect_padding(device, dtype) and supports_replicate_padding(device, dtype)):
            pytest.skip("torch has no reflect or replicate padding kernel for this device and dtype")

    @staticmethod
    def _ramped_step(device, dtype):
        # 0 | 0.6 | 1 across x = 6..8 of a 9x14 image; with no blur the raw Sobel |gx| is 2.4, 4, 1.6 at x = 6, 7, 8,
        # so the ridge at x = 7 is a strict maximum
        img = torch.zeros(1, 1, 9, 14, device=device, dtype=dtype)
        img[..., 7] = 0.6
        img[..., 8:] = 1.0
        return img

    @staticmethod
    def _diagonal_lines(device):
        # u = x + y on a 12x17 grid: the lines u = const run at 45 degrees and the gradient across them along (1, 1)
        ys, xs = torch.meshgrid(torch.arange(12, device=device), torch.arange(17, device=device), indexing="ij")
        return xs + ys

    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("dark_to_bright", [True, False])
    def test_two_level_step_keeps_one_pixel_of_the_tie(self, transpose, dark_to_bright, device, dtype):
        self._require_padding(device, dtype)
        # a step between x = 6 and x = 7 of a 9x14 image, no blur: the raw Sobel magnitudes of x = 6 and x = 7 are
        # both 4. As in OpenCV's Canny, a pixel must be strictly greater than its left (upper) neighbour and greater
        # than or equal to its right (lower) one, so the left (upper) pixel of the tie is kept, for either polarity
        img = torch.zeros(1, 1, 9, 14, device=device, dtype=dtype)
        img[..., 7:] = 1.0
        if not dark_to_bright:
            img = 1.0 - img
        expected = torch.zeros_like(img)
        expected[..., 6] = 1.0
        if transpose:
            img, expected = img.transpose(-1, -2), expected.transpose(-1, -2)
        magnitude, edges = canny(img, kernel_size=1)
        self.assert_close(edges, expected)
        self.assert_close(magnitude, 4.0 * expected)

    def test_square_without_blur_matches_opencv(self, device, dtype):
        self._require_padding(device, dtype)
        # a 10x10 square at rows and columns 10..19 of a 40x40 image, no blur: every side is a two-level step whose
        # tied pair keeps its left or upper pixel, outside the square on the left and top, inside on the right and
        # bottom. The suppression compares magnitudes exactly, so no rounding decides a tie. OpenCV 5.0.0 gives the same
        # map: cv2.Canny((255 * img).astype(np.uint8), 25.5, 51, L2gradient=True)
        img = torch.zeros(1, 1, 40, 40, device=device, dtype=dtype)
        img[..., 10:20, 10:20] = 1.0
        expected = torch.zeros_like(img)
        expected[..., 9, 11:19] = 1.0  # top
        expected[..., 19, 10:20] = 1.0  # bottom
        expected[..., 11:19, 9] = 1.0  # left
        expected[..., 10:19, 19] = 1.0  # right
        expected[..., 10, 10] = 1.0  # top-left corner
        _, edges = canny(img, kernel_size=1)
        self.assert_close(edges, expected)

    def test_square_contour_is_closed(self, device, dtype):
        self._require_padding(device, dtype)
        # a 10x10 square at rows and columns 25..34 of a 100x100 image, default arguments: after the 5x5 blur the two
        # pixels across each side tie in exact arithmetic and rounding decides which one is larger. Exactly one of the
        # pair survives either way, so every row and column along a side holds one edge pixel on each side of the square
        img = torch.zeros(1, 1, 100, 100, device=device, dtype=dtype)
        img[..., 25:35, 25:35] = 1.0
        _, edges = canny(img)
        on = edges[0, 0] > 0
        for k in range(27, 33):
            for line in (on[k], on[:, k]):
                assert line[23:27].sum().item() == 1
                assert line[33:37].sum().item() == 1
                assert line.sum().item() == 2

    @pytest.mark.parametrize("mirror", [False, True])
    def test_two_level_diagonal_step_is_one_pixel_wide_along_the_gradient(self, mirror, device, dtype):
        self._require_padding(device, dtype)
        # guard: holds on main too; fails if the diagonal neighbours are taken along another direction
        # 0 | 1 across x + y = 12.5 of a 12x17 image, no blur: the raw Sobel components are 1, 3, 3, 1 on the lines
        # x + y = 11..14. The tied lines 12 and 13 are 4-neighbours across the gradient, not neighbours along it; along
        # (1, 1) each pixel is compared with the lines two steps away, so both survive, as in OpenCV, and every line
        # along the gradient crosses the edge once. Mirrored in x, the same holds along (-1, 1)
        u = self._diagonal_lines(device)
        img = (u >= 13).to(dtype)[None, None]
        expected = ((u == 12) | (u == 13)).to(dtype)[None, None]
        if mirror:
            img, expected = img.flip(-1), expected.flip(-1)
        _, edges = canny(img, kernel_size=1)
        # rows 1..10: the replicated top and bottom borders change the gradient on rows 0 and 11
        self.assert_close(edges[..., 1:-1, :], expected[..., 1:-1, :])

    @pytest.mark.parametrize("mirror", [False, True])
    def test_diagonal_tie_keeps_the_pixel_between(self, mirror, device, dtype):
        self._require_padding(device, dtype)
        # guard: holds on main too; fails if a diagonal comparison accepts a tie
        # 0 | 0.5 | 1 on x + y < 13, = 13, > 13, no blur: the raw Sobel components are 2, 3, 2 on the lines 12, 13, 14,
        # so the lines 12 and 14 tie as neighbours along the (1, 1) gradient. Along a diagonal both comparisons stay
        # strict, as in OpenCV: the tie keeps neither, and the line 13 between them carries a one-pixel-wide edge
        u = self._diagonal_lines(device)
        img = torch.where(u < 13, 0.0, torch.where(u == 13, 0.5, 1.0)).to(dtype)[None, None]
        expected = (u == 13).to(dtype)[None, None]
        if mirror:
            img, expected = img.flip(-1), expected.flip(-1)
        _, edges = canny(img, kernel_size=1)
        self.assert_close(edges[..., 1:-1, :], expected[..., 1:-1, :])

    @pytest.mark.parametrize("transpose", [False, True])
    def test_ridge_without_tie_keeps_its_maximum(self, transpose, device, dtype):
        self._require_padding(device, dtype)
        # guard: holds on main too; fails if the suppression reads a zero border instead of a flat one
        # the strict maximum at x = 7 is the edge; the flat regions have no gradient and are never edges, also at the
        # image border
        img = self._ramped_step(device, dtype)
        expected = torch.zeros_like(img)
        expected[..., 7] = 1.0
        if transpose:
            img, expected = img.transpose(-1, -2), expected.transpose(-1, -2)
        magnitude, edges = canny(img, kernel_size=1)
        self.assert_close(edges, expected)
        self.assert_close(magnitude, 4.0 * expected)

    def test_flat_image_has_no_edges_at_any_threshold(self, device, dtype):
        self._require_padding(device, dtype)
        # guard: holds on main too; fails if the suppression reads a zero border instead of a flat one
        # a flat region has magnitude sqrt(eps) = 1e-3, above a threshold of 1e-4, but it ties with its neighbours and
        # with the image border, which has no gradient either, so no pixel of it is a maximum
        img = torch.full((1, 1, 9, 14), 0.3, device=device, dtype=dtype)
        magnitude, edges = canny(img, 1e-4, 2e-4, kernel_size=1)
        self.assert_close(magnitude, torch.zeros_like(img))
        self.assert_close(edges, torch.zeros_like(img))

    def test_thresholds_are_in_unnormalized_sobel_units(self, device, dtype):
        self._require_padding(device, dtype)
        # the ridge at x = 7 has magnitude 4, eight times what sobel() returns by default, and the thresholds compare
        # against it, so values above 1 are meaningful
        img = self._ramped_step(device, dtype)
        self.assert_close(sobel(img)[..., 7], torch.full((1, 1, 9), 0.5, device=device, dtype=dtype))
        ridge = torch.zeros_like(img)
        ridge[..., 7] = 1.0
        for low, high, expected in ((3.0, 3.5, ridge), (3.0, 4.5, 0.5 * ridge), (4.5, 5.0, 0.0 * ridge)):
            _, edges = canny(img, low, high, kernel_size=1, hysteresis=False)
            self.assert_close(edges, expected)
            _, edges = Canny(low, high, kernel_size=1, hysteresis=False)(img)
            self.assert_close(edges, expected)

    @pytest.mark.device_agnostic
    def test_onnx_export_legacy_matches_eager(self, dtype):
        if dtype != torch.float32:
            pytest.skip("the exported graph is checked once, in float32")
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        # the legacy exporter's atan2 is NaN at (0, 0); cast to an index, NaN gives 0 on arm64 but an out-of-range
        # value on x86-64, where GatherElements raises. The flat image and the step's flat columns have zero gradient,
        # and they must run and not be maxima, while the step keeps its left pixel
        step = torch.zeros(1, 1, 9, 14, dtype=dtype)
        step[..., 7:] = 1.0
        flat = torch.full_like(step, 0.3)
        model = Canny(kernel_size=1, hysteresis=False)
        buffer = io.BytesIO()
        export_kwargs: dict[str, Any] = {"input_names": ["input"], "opset_version": 17}
        if torch_version_ge(2, 5, 0):
            export_kwargs["dynamo"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, step, buffer, **export_kwargs)
        session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
        for img in (step, flat):
            magnitude, edges = (torch.from_numpy(out) for out in session.run(None, {"input": img.numpy()}))
            expected_magnitude, expected_edges = model(img)
            self.assert_close(magnitude, expected_magnitude)
            self.assert_close(edges, expected_edges)

    @pytest.mark.parametrize("channels", [2, 4])
    def test_unsupported_channel_count_raises(self, channels, device, dtype):
        with pytest.raises(ImageError, match="1 or 3 channels"):
            canny(torch.zeros(1, channels, 12, 13, device=device, dtype=dtype))


class TestConventionsCanny(BaseTester):
    @staticmethod
    def _require_padding(device, dtype):
        # the Gaussian blur pads by reflection, the Sobel gradient by replication
        if not (supports_reflect_padding(device, dtype) and supports_replicate_padding(device, dtype)):
            pytest.skip("torch has no reflect or replicate padding kernel for this device and dtype")

    @staticmethod
    def _ramped_step(height, device, dtype):
        # 0 | 0.6 * height | height across x = 6..8 of a 9x14 image; with kernel_size=1 (no blur) the raw Sobel |gx|
        # is 2.4h, 4h and 1.6h at x = 6, 7, 8, so the ridge at x = 7 has no tie
        img = torch.zeros(1, 1, 9, 14, device=device, dtype=dtype)
        img[..., 7] = 0.6 * height
        img[..., 8:] = height
        return img

    def test_convention_canny_magnitude_is_unnormalized_sobel_after_nms(self, device, dtype):
        self._require_padding(device, dtype)
        img = self._ramped_step(0.125, device, dtype)
        magnitude, edges = canny(img, 0.4, 0.49, kernel_size=1, eps=0.0)
        assert magnitude.dtype == edges.dtype == dtype
        # the thresholds compare against the unnormalized Sobel magnitude 4h = 0.5, eight times sobel()'s default
        self.assert_close(sobel(img, eps=0.0)[..., 7], torch.full((1, 1, 9), 0.0625, device=device, dtype=dtype))
        expected = torch.zeros_like(img)
        expected[..., 7] = 0.5  # returned after non-maximum suppression: zero off the ridge
        self.assert_close(magnitude, expected)
        self.assert_close(edges, expected * 2)
        # relabel: the transposed step gives the transposed result
        magnitude_t, edges_t = canny(img.transpose(-1, -2), 0.4, 0.49, kernel_size=1, eps=0.0)
        self.assert_close(magnitude_t, magnitude.transpose(-1, -2))
        self.assert_close(edges_t, edges.transpose(-1, -2))
        # with the default 5x5, sigma=1 blur: the Sobel of gaussian_blur2d(img), unnormalized, eps inside the root
        grad = spatial_gradient(gaussian_blur2d(img, (5, 5), (1.0, 1.0)), normalized=False)
        reference = torch.sqrt(grad[:, :, 0] ** 2 + grad[:, :, 1] ** 2 + 1e-6)
        magnitude, _ = canny(img)
        kept = magnitude > 0.1  # the ridge (about 0.32), not the sqrt(eps) floor of the flat regions
        assert kept.sum() == 9  # one ridge pixel per row
        self.assert_close(magnitude[kept], reference[kept])

    def test_convention_canny_thresholds_are_strict(self, device, dtype):
        self._require_padding(device, dtype)
        # the ridge magnitude is exactly 0.5 (eps=0); hysteresis=False keeps the weak (0.5) / strong (1) labels
        img = self._ramped_step(0.125, device, dtype)
        ridge = (0, 0, 4, 7)
        _, edges = canny(img, 0.4, 0.5, kernel_size=1, eps=0.0, hysteresis=False)
        assert edges[ridge].item() == 0.5  # equal to high_threshold: weak, not strong
        _, edges = canny(img, 0.5, 0.5, kernel_size=1, eps=0.0, hysteresis=False)  # low == high is accepted
        assert edges.sum().item() == 0  # equal to low_threshold: dropped
        _, edges = canny(img, 0.4, 0.49, kernel_size=1, eps=0.0, hysteresis=False)
        assert edges[ridge].item() == 1.0

    def test_convention_canny_converts_rgb_to_grayscale(self, device, dtype):
        self._require_padding(device, dtype)
        generator = torch.Generator().manual_seed(0)
        rgb = torch.rand(2, 3, 11, 13, generator=generator).to(device=device, dtype=dtype)
        magnitude, edges = canny(rgb)
        assert magnitude.shape == edges.shape == (2, 1, 11, 13)
        magnitude_grey, edges_grey = canny(rgb_to_grayscale(rgb))
        self.assert_close(magnitude, magnitude_grey)
        self.assert_close(edges, edges_grey)
        # channel 0 is read as red: the same data in BGR order gives other edges
        _, edges_bgr = canny(rgb.flip(1))
        assert not torch.equal(edges_bgr, edges)

    def test_convention_canny_hysteresis_is_8_connected(self, device, dtype):
        self._require_padding(device, dtype)
        # hysteresis keeps a weak pixel connected to a strong one through any of its 8 neighbours, iterated to
        # convergence; the reference grows the strong set from canny's own weak / strong labels
        generator = torch.Generator().manual_seed(2)
        img = torch.rand(1, 1, 12, 17, generator=generator).to(device=device, dtype=dtype)
        _, labels = canny(img, 0.3, 0.6, hysteresis=False)
        _, edges = canny(img, 0.3, 0.6, hysteresis=True)
        weak, strong = (labels == 0.5).float(), (labels == 1).float()
        cross = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], device=device)

        def grow(dilate):
            kept, steps = strong, 0
            while True:
                grown = torch.maximum(strong, weak * (dilate(kept) > 0).float())
                if torch.equal(grown, kept):
                    return kept, steps
                kept, steps = grown, steps + 1

        kept_8, steps_8 = grow(lambda k: F.max_pool2d(k, 3, 1, 1))
        kept_4, _ = grow(lambda k: F.conv2d(k, cross, padding=1))
        assert steps_8 > 1  # the fixture needs more than one pass
        assert not torch.equal(kept_8, kept_4)  # and separates the two connectivities
        self.assert_close(edges, kept_8.to(dtype))

    def test_wart_canny_two_level_step_has_no_edge_5170(self, device, dtype):
        """#5170: NMS needs strictly greater on both sides, so the tied pair of a two-level step is suppressed."""
        self._require_padding(device, dtype)
        step = torch.zeros(1, 1, 9, 14, device=device, dtype=dtype)
        step[..., 7:] = 1.0
        for img in (step, step.transpose(-1, -2)):
            _, edges = canny(img, kernel_size=1)
            assert edges.sum().item() == 0
        # control: the ramped step has no tie and gives one pixel per row
        _, edges = canny(self._ramped_step(1.0, device, dtype), kernel_size=1)
        assert edges[0, 0].nonzero()[:, 1].tolist() == [7] * 9

    def test_wart_canny_rejects_thresholds_of_one_and_above_5171(self, device, dtype):
        """#5171: thresholds must lie in (0, 1) though the magnitude of a [0, 1] image reaches 4."""
        self._require_padding(device, dtype)
        img = self._ramped_step(1.0, device, dtype)
        magnitude, _ = canny(img, kernel_size=1)
        self.assert_close(magnitude.max(), torch.tensor(4.0, device=device, dtype=dtype))
        with pytest.raises(BaseError):
            canny(img, 2.0, 3.0, kernel_size=1)
        with pytest.raises(BaseError):
            Canny(2.0, 3.0)

    def test_wart_canny_unsupported_channel_count_is_not_validated_5171(self, device, dtype):
        """#5171: only C in {1, 3} works; other channel counts fail inside torch, not in a kornia check."""
        self._require_padding(device, dtype)  # else the missing pad kernel raises first
        for channels in (2, 4):
            with pytest.raises(Exception) as excinfo:
                canny(torch.rand(1, channels, 12, 13, device=device, dtype=dtype))
            assert not isinstance(excinfo.value, BaseError)
