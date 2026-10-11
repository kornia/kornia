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

import pytest
import torch
import torch.nn.functional as F

from kornia.core.check import ShapeError
from kornia.feature.sold2 import SOLD2, SOLD2_detector
from kornia.feature.sold2.sold2 import keypoints_to_grid
from kornia.feature.sold2.sold2_detector import LineSegmentDetectionModule, line_map_to_segments, prob_to_junctions
from kornia.feature.sold2.structures import LineDetectorCfg

from testing.base import BaseTester


class TestSOLD2_detector(BaseTester):
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_shape(self, device, batch_size, dtype):
        inp = torch.ones(batch_size, 1, 64, 64, device=device, dtype=dtype)
        sold2 = SOLD2_detector(pretrained=False).to(device, dtype)
        out = sold2(inp)
        assert out["junction_heatmap"].shape == (batch_size, 64, 64)
        assert out["line_heatmap"].shape == (batch_size, 64, 64)

    @pytest.mark.skip("Takes ages to run")
    def test_gradcheck(self, device):
        img = torch.rand(2, 1, 128, 128, device=device, dtype=torch.float64)
        sold2 = SOLD2_detector(pretrained=False).to(img.device, img.dtype)

        def proxy_forward(x):
            return sold2.forward(x)["junction_heatmap"]

        self.gradcheck(proxy_forward, (img,), eps=1e-4, atol=1e-4)

    @pytest.mark.skip("Does not like recursive definition of Hourglass in backbones.py l.134.")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 128, 128
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        # pretrained=False: scripting does not need the weights, and the CI cache
        # deliberately does not carry ``sold2_wireframe.tar`` -- see
        # ``NOT_PREFETCHED`` in tests/core/test_weights_prefetch.py.
        model = SOLD2_detector(pretrained=False).to(img.device, img.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(img), model_jit(img))


class TestJunctionsAndSegments(BaseTester):
    @staticmethod
    def _junction_map(points, device, dtype, shape=(96, 128)):
        prob = torch.zeros(shape, device=device, dtype=dtype)
        for (row, col), score in points:
            prob[row, col] = score
        return prob

    @pytest.mark.parametrize(
        "offset, n_kept",
        [
            # (d_row, d_col) between the strong and the weak junction; the reference suppresses up to and at 8 px
            ((0, 7), 1),
            ((0, 8), 1),
            ((0, 9), 2),
            ((8, 0), 1),
            ((9, 0), 2),
            ((8, 8), 1),
            ((8, 9), 2),
            ((-8, 3), 1),
            ((-9, 3), 2),
        ],
    )
    def test_nms_radius_matches_reference_NNNN(self, device, dtype, offset, n_kept):
        # reference: cvg/SOLD2 sold2/model/metrics.py ``super_nms`` / ``nms_fast`` (pad = 8, inclusive window)
        r0, c0 = 40, 30
        prob = self._junction_map([((r0, c0), 1.0), ((r0 + offset[0], c0 + offset[1]), 0.9)], device, dtype)
        junctions = prob_to_junctions(prob, 8, 0.0153846, 500)
        assert len(junctions) == n_kept
        assert [float(r0), float(c0)] in junctions.tolist()

    def test_nms_is_greedy_by_score_NNNN(self, device, dtype):
        # the strongest junction suppresses both of its 8 px neighbours, which cannot then suppress the one beyond
        prob = self._junction_map([((40, 20), 0.9), ((40, 28), 1.0), ((40, 37), 0.8)], device, dtype)
        junctions = prob_to_junctions(prob, 8, 0.0153846, 500)
        assert junctions.tolist() == [[40.0, 28.0], [40.0, 37.0]]

    def test_junctions_are_in_raster_order_NNNN(self, device, dtype):
        # scores run against the raster order, in an H != W map
        points = [((70, 100), 1.0), ((30, 20), 0.9), ((30, 60), 0.8), ((10, 110), 0.7), ((90, 5), 0.6)]
        prob = self._junction_map(points, device, dtype)
        junctions = prob_to_junctions(prob, 8, 0.0153846, 500)
        expected = [[10.0, 110.0], [30.0, 20.0], [30.0, 60.0], [70.0, 100.0], [90.0, 5.0]]
        assert junctions.tolist() == expected

    def test_top_k_selects_by_score_then_raster_order_NNNN(self, device, dtype):
        points = [((70, 100), 1.0), ((30, 20), 0.9), ((30, 60), 0.8), ((10, 110), 0.7)]
        prob = self._junction_map(points, device, dtype)
        junctions = prob_to_junctions(prob, 8, 0.0153846, 3)
        assert junctions.tolist() == [[30.0, 20.0], [30.0, 60.0], [70.0, 100.0]]

    @pytest.mark.parametrize("use_junction_refinement", [False, True])
    def test_segment_endpoints_are_in_raster_order_NNNN(self, device, dtype, use_junction_refinement):
        # a line from (x, y) = (20, 30) to (100, 70), with the bottom-right junction the stronger one
        prob = self._junction_map([((70, 100), 1.0), ((30, 20), 0.5)], device, dtype)
        heatmap = torch.zeros(96, 128, device=device, dtype=dtype)
        t = torch.linspace(0, 1, 400, device=device)
        rows = torch.round(30 + (70 - 30) * t).long()
        cols = torch.round(20 + (100 - 20) * t).long()
        heatmap[rows, cols] = 1.0
        cfg = LineDetectorCfg(
            use_heatmap_refinement=False,
            use_junction_refinement=use_junction_refinement,
            use_candidate_suppression=False,
        )
        junctions = prob_to_junctions(prob, 8, 0.0153846, 500)
        line_map, junctions, _ = LineSegmentDetectionModule(cfg).detect(junctions, heatmap)
        segments = line_map_to_segments(junctions, line_map)
        assert segments.shape == (1, 2, 2)
        assert segments[0, 0, 0] < segments[0, 1, 0]
        expected = torch.tensor([[30.0, 20.0], [70.0, 100.0]], device=device, dtype=segments.dtype)
        self.assert_close(segments[0], expected, atol=1.0, rtol=0)


class TestSOLD2(BaseTester):
    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_shape(self, device, batch_size, dtype):
        inp = torch.ones(batch_size, 1, 64, 64, device=device, dtype=dtype)
        sold2 = SOLD2(pretrained=False).to(device, dtype)
        out = sold2(inp)
        assert out["dense_desc"].shape == (batch_size, 128, 16, 16)

    @pytest.mark.skip("Takes ages to run")
    def test_gradcheck(self, device):
        img = torch.rand(2, 1, 256, 256, device=device, dtype=torch.float64)
        sold2 = SOLD2(pretrained=False).to(img.device, img.dtype)

        def proxy_forward(x):
            return sold2.forward(x)["dense_desc"]

        self.gradcheck(proxy_forward, (img,), eps=1e-4, atol=1e-4)

    @pytest.mark.skip("Does not like recursive definition of Hourglass in backbones.py l.134.")
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 256, 256
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SOLD2().to(img.device, img.dtype).eval()
        model_jit = torch.jit.script(model)
        self.assert_close(model(img), model_jit(img))


class TestKeypointsToGrid(BaseTester):
    # keypoints_to_grid feeds F.grid_sample(..., align_corners=False) in WunschLineMatcher, so its
    # normalization has to be the one the reference implementation (cvg/SOLD2,
    # sold2/misc/geometry_utils.py) trained the weights with: keypoints * 2 / img_size - 1. Until
    # #4554 it used the corner-aligned normalize_pixel_coordinates, which drifts by p / (S - 1)
    # image pixels toward the far edge under that sampler.

    def test_smoke(self, device, dtype):
        kp = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device, dtype=dtype)
        grid = keypoints_to_grid(kp, (32, 64))
        assert grid.shape == (1, 3, 1, 2)
        assert grid.dtype == dtype
        assert grid.device == kp.device

    def test_exception(self, device, dtype):
        with pytest.raises(ShapeError, match=r"Expected shape: \['N', '2'\]"):
            keypoints_to_grid(torch.zeros(3, 3, device=device, dtype=dtype), (32, 64))
        with pytest.raises(ShapeError, match=r"Expected shape: \['N', '2'\]"):
            keypoints_to_grid(torch.zeros(1, 3, 2, device=device, dtype=dtype), (32, 64))

    def test_integer_keypoints_follow_the_default_dtype(self, device):
        # Integer keypoints promote through the float arithmetic to the ambient default dtype, the
        # dtype grid_sample will expect for a descriptor map built under the same default, and the
        # values are the reference ones (0 -> -1, the half size -> 0).
        kp = torch.tensor([[0, 0], [16, 32]], device=device, dtype=torch.int64)
        grid = keypoints_to_grid(kp, (32, 64))
        assert grid.dtype == torch.get_default_dtype()
        self.assert_close(grid[0, :, 0], torch.tensor([[-1.0, -1.0], [0.0, 0.0]], device=device))

    def test_convention_matches_reference_normalization_4554(self, device, dtype):
        # Reference: cvg/SOLD2 sold2/misc/geometry_utils.py, keypoints_to_grid.
        #   grid = keypoints.float() * 2 / torch.tensor(img_size) - 1, then (x, y) reorder.
        # The figures are the issue's: on a 480x640 image with grid_size 8 (a 60x80 descriptor
        # map), grid_sample(align_corners=False) reads descriptor coordinate (g + 1) * S_desc / 2 - 0.5.
        H, W, g = 480, 640, 8
        kp = torch.tensor([[0.0, 0.0], [240.0, 320.0], [479.0, 639.0], [100.0, 639.0]], device=device, dtype=dtype)
        grid = keypoints_to_grid(kp, (H, W))

        reference = kp * 2.0 / torch.tensor([H, W], device=device, dtype=dtype) - 1.0
        reference = reference[:, [1, 0]].view(1, -1, 1, 2)
        self.assert_close(grid, reference, atol=0.0, rtol=0.0)

        desc_size = torch.tensor([W // g, H // g], device=device, dtype=dtype)  # (x, y)
        reads = (grid[0, :, 0] + 1.0) * desc_size / 2.0 - 0.5
        expected = torch.tensor(
            [[-0.5, -0.5], [39.5, 29.5], [79.375, 59.375], [79.375, 12.0]], device=device, dtype=dtype
        )
        # Pre-#4554 the same keypoints read (-0.5, -0.5), (39.5625, 29.5625), (79.5, 59.5),
        # (79.5, 12.026), i.e. half a descriptor pixel past the last centre at the far edge.
        self.assert_close(reads, expected)

    def test_convention_cell_keypoint_reads_its_own_descriptor_4554(self, device, dtype):
        # Under the reference normalization image pixel p reads descriptor coordinate p / g - 0.5, so
        # the keypoint (g * (i + 0.5), g * (j + 0.5)) lands exactly on descriptor (i, j) and sampling
        # with the matcher's align_corners=False returns that descriptor untouched. (That keypoint is
        # half an image pixel past the centre of cell (i, j) under kornia's pixel-centre convention,
        # which is the constant offset from the half-pixel mapping the docstring records.)
        # Power-of-two sizes keep every coordinate exact at half precision.
        H, W, g = 32, 64, 8
        hd, wd = H // g, W // g
        desc = torch.rand(1, 5, hd, wd, device=device, dtype=dtype)
        ii, jj = torch.meshgrid(torch.arange(hd, device=device), torch.arange(wd, device=device), indexing="ij")
        kp = torch.stack(((ii.flatten() + 0.5) * g, (jj.flatten() + 0.5) * g), dim=-1).to(dtype)
        grid = keypoints_to_grid(kp, (H, W))
        sampled = F.grid_sample(desc, grid, align_corners=False)[0, :, :, 0]  # (C, hd * wd)
        self.assert_close(sampled, desc[0].reshape(5, -1))

    def test_dynamo(self, device, dtype, torch_optimizer):
        kp = torch.tensor([[1.0, 2.0], [30.0, 60.0]], device=device, dtype=dtype)
        op = keypoints_to_grid
        op_optimized = torch_optimizer(op)
        self.assert_close(op(kp, (32, 64)), op_optimized(kp, (32, 64)))
