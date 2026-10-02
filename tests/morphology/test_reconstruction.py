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

import math

import pytest
import torch

from kornia.core.check import are_checks_enabled, disable_checks, enable_checks
from kornia.core.exceptions import BaseError, ShapeError, TypeCheckError, ValueCheckError
from kornia.morphology import reconstruction

from testing.base import BaseTester
from testing.parametrized_tester import parametrized_test

# Generated with scikit-image 0.26.0 / numpy 2.5.3:
#   seed = np.minimum(MASK, 0.3); seed[2, 2] = 0.9
#   reconstruction(seed, MASK)                                          -> DILATION_SQUARE
#   reconstruction(seed, MASK, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])) -> DILATION_CROSS
#   eseed = np.maximum(MASK, 0.6); eseed[0, 3] = 0.8
#   reconstruction(eseed, MASK, method="erosion")                       -> EROSION_SQUARE
MASK = [[0.2, 0.9, 0.1, 0.8, 0.3], [0.4, 0.2, 0.2, 0.7, 0.6], [0.1, 0.5, 0.9, 0.1, 0.4], [0.7, 0.3, 0.6, 0.2, 0.9]]
DILATION_SQUARE = [
    [0.2, 0.4, 0.1, 0.7, 0.3],
    [0.4, 0.2, 0.2, 0.7, 0.6],
    [0.1, 0.5, 0.9, 0.1, 0.4],
    [0.5, 0.3, 0.6, 0.2, 0.4],
]
DILATION_CROSS = [
    [0.2, 0.3, 0.1, 0.3, 0.3],
    [0.3, 0.2, 0.2, 0.3, 0.3],
    [0.1, 0.5, 0.9, 0.1, 0.3],
    [0.3, 0.3, 0.6, 0.2, 0.3],
]
EROSION_SQUARE = [
    [0.6, 0.9, 0.6, 0.8, 0.6],
    [0.6, 0.6, 0.6, 0.7, 0.6],
    [0.6, 0.6, 0.9, 0.6, 0.6],
    [0.7, 0.6, 0.6, 0.6, 0.9],
]


def _serpentine(n, device, dtype):
    # One corridor that zigzags through every other row, so its path length is about n^2 / 2.
    mask = torch.zeros(1, 1, n, n, device=device, dtype=dtype)
    mask[..., ::2, :] = 1.0
    for i, row in enumerate(range(1, n, 2)):
        mask[..., row, n - 1 if i % 2 == 0 else 0] = 1.0
    return mask


def _distinct(*shape, device):
    # Values at least 1 / numel apart, so gradcheck's finite differences never cross a min/max kink.
    n = math.prod(shape)
    values = torch.randperm(n, generator=torch.Generator().manual_seed(0)).double() / n
    return values.view(shape).to(device)


@parametrized_test(
    smoke_inputs=lambda device, dtype: (
        torch.rand(1, 3, 4, 4, device=device, dtype=dtype) * 0.5,
        torch.rand(1, 3, 4, 4, device=device, dtype=dtype),
    ),
    cardinality_tests=[
        {
            "inputs": lambda device, dtype: (
                torch.zeros(2, 3, 2, 4, device=device, dtype=dtype),
                torch.ones(2, 3, 2, 4, device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([2, 3, 2, 4]),
        },
        {
            "inputs": lambda device, dtype: (
                torch.zeros(3, 2, 5, 1, device=device, dtype=dtype),
                torch.ones(3, 2, 5, 1, device=device, dtype=dtype),
            ),
            "expected_shape": torch.Size([3, 2, 5, 1]),
        },
    ],
    gradcheck_inputs=lambda device: tuple(t.clone().requires_grad_() for t in _distinct(2, 2, 2, 5, 5, device=device)),
)
class TestReconstruction(BaseTester):
    def setup_method(self) -> None:
        self.func = reconstruction

    def test_dilation(self, device, dtype):
        mask = torch.tensor(MASK, device=device, dtype=dtype)[None, None]
        seed = mask.clamp(max=0.3)
        seed[..., 2, 2] = mask[..., 2, 2]
        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)

        square_expected = torch.tensor(DILATION_SQUARE, device=device, dtype=dtype)[None, None]
        cross_expected = torch.tensor(DILATION_CROSS, device=device, dtype=dtype)[None, None]

        self.assert_close(reconstruction(seed, mask), square_expected)
        self.assert_close(reconstruction(seed, mask, cross), cross_expected)

    def test_erosion(self, device, dtype):
        mask = torch.tensor(MASK, device=device, dtype=dtype)[None, None]
        seed = mask.clamp(min=0.6)
        seed[..., 0, 3] = mask[..., 0, 3]

        self.assert_close(
            reconstruction(seed, mask, method="erosion"),
            torch.tensor(EROSION_SQUARE, device=device, dtype=dtype)[None, None],
        )

    @pytest.mark.parametrize("engine", ["unfold", "shift"])
    def test_converges_past_image_size(self, device, dtype, engine):
        # A 31 x 31 serpentine needs hundreds of steps, far more than max(H, W).
        mask = _serpentine(31, device, dtype)
        seed = torch.zeros_like(mask)
        seed[..., 0, 0] = 1.0

        assert torch.equal(reconstruction(seed, mask, engine=engine), mask)
        assert not torch.equal(reconstruction(seed, mask, num_iters=31, engine=engine), mask)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("engine", ["auto", "unfold", "shift"])
    @pytest.mark.parametrize("method", ["dilation", "erosion"])
    def test_converges_on_random_data(self, device, dtype, engine, method):
        # Non-binary values expose an inexact step, which can oscillate and never converge. A path through a
        # 12 x 12 image has at most 143 steps, so a fixed 144 steps is the converged result.
        mask = torch.rand(2, 3, 12, 12, device=device, dtype=dtype)
        seed = mask * torch.rand_like(mask) if method == "dilation" else mask + torch.rand_like(mask)

        expected = reconstruction(seed, mask, method=method, num_iters=144, engine=engine)
        assert torch.equal(reconstruction(seed, mask, method=method, engine=engine), expected)

    def test_asymmetric_kernel_direction(self, device, dtype):
        # Both methods spread a value to the kernel's offsets, as `dilation` does. Generated with scikit-image 0.26.0:
        #   m = np.full((1, 5), 3.0); s = np.zeros((1, 5)); s[0, 2] = 3; e = np.full((1, 5), 9.0); e[0, 2] = 3
        #   reconstruction(s, m, footprint=np.array([[0, 1, 1]]))                   -> [[0, 0, 3, 3, 3]]
        #   reconstruction(e, m, method="erosion", footprint=np.array([[0, 1, 1]])) -> [[9, 9, 3, 3, 3]]
        kernel = torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype)
        mask = torch.full((1, 1, 1, 5), 3.0, device=device, dtype=dtype)
        seed = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 1, 5)
        eseed = torch.tensor([9.0, 9.0, 3.0, 9.0, 9.0], device=device, dtype=dtype).view(1, 1, 1, 5)

        assert reconstruction(seed, mask, kernel).flatten().tolist() == [0.0, 0.0, 3.0, 3.0, 3.0]
        assert reconstruction(eseed, mask, kernel, method="erosion").flatten().tolist() == [9.0, 9.0, 3.0, 3.0, 3.0]

    def test_center_always_included(self, device, dtype):
        # scikit-image keeps each pixel in its own neighborhood even when the footprint leaves the center out.
        ring = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        mask = torch.rand(1, 1, 6, 6, device=device, dtype=dtype)
        seed = mask * 0.5

        assert torch.equal(reconstruction(seed, mask, ring), reconstruction(seed, mask))

    def test_seed_above_mask_is_clipped(self, device, dtype):
        mask = torch.rand(1, 2, 5, 5, device=device, dtype=dtype)
        seed = mask + 1.0

        assert torch.equal(reconstruction(seed, mask), mask)
        assert torch.equal(reconstruction(mask - 1.0, mask, method="erosion"), mask)

    def test_num_iters(self, device, dtype):
        mask = _serpentine(9, device, dtype)
        seed = torch.zeros_like(mask)
        seed[..., 0, 0] = 1.0

        assert torch.equal(reconstruction(seed, mask, num_iters=0), seed)
        assert torch.equal(reconstruction(seed, mask, num_iters=1000), reconstruction(seed, mask))

    def test_num_iters_runs_exactly(self, device, dtype):
        # Each step spreads the value one pixel, so the count shows in how far it got.
        mask = torch.ones(1, 1, 1, 7, device=device, dtype=dtype)
        seed = torch.zeros_like(mask)
        seed[..., 0] = 1.0

        dilated = reconstruction(seed, mask, num_iters=2)
        eroded = reconstruction(1 - seed, 1 - mask, method="erosion", num_iters=2)

        assert dilated.flatten().tolist() == [1, 1, 1, 0, 0, 0, 0]
        assert eroded.flatten().tolist() == [0, 0, 0, 1, 1, 1, 1]

    @pytest.mark.parametrize("check_every", [1, 3, 64])
    def test_check_every(self, device, dtype, check_every):
        mask = _serpentine(9, device, dtype)
        seed = torch.zeros_like(mask)
        seed[..., 0, 0] = 1.0

        assert torch.equal(reconstruction(seed, mask, check_every=check_every), mask)
        assert torch.equal(reconstruction(1 - seed, 1 - mask, method="erosion", check_every=check_every), 1 - mask)

    def test_kernel_zeros_exclude_neighbors(self, device, dtype):
        # A kernel's zero cells are excluded, so the diagonal 5e4 never reaches the unconnected 3e4 pixel.
        # scikit-image leaves that pixel at 0.
        mask = torch.tensor([[5e4, 0.0], [0.0, 3e4]], device=device, dtype=dtype)[None, None]
        seed = torch.zeros_like(mask)
        seed[..., 0, 0] = mask[..., 0, 0]
        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)

        assert torch.equal(reconstruction(seed, mask, cross), seed)

    @pytest.mark.parametrize("kernel_dtype", [torch.bool, torch.uint8, torch.int64, torch.float32])
    def test_kernel_dtype_is_ignored(self, device, dtype, kernel_dtype):
        # The kernel is only a membership mask: its dtype changes neither the result nor the output dtype. A
        # float32 kernel, such as the default dtype of `torch.ones(3, 3)`, must not widen half-precision inputs.
        mask = torch.rand(2, 3, 6, 7, device=device, dtype=dtype)
        seed = mask * 0.5
        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = reconstruction(seed, mask, cross)

        actual = reconstruction(seed, mask, cross.to(kernel_dtype))
        assert actual.dtype == dtype
        assert torch.equal(actual, expected)

    def test_gradcheck_erosion(self, device):
        seed, mask = _distinct(2, 2, 2, 5, 5, device=device)

        self.gradcheck(lambda s, m: reconstruction(s, m, method="erosion"), (seed, mask))

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)

        with pytest.raises(TypeCheckError):
            reconstruction([0.0], tensor)

        with pytest.raises(TypeCheckError):
            reconstruction(tensor, [0.0])

        with pytest.raises(BaseError, match="floating-point"):
            reconstruction(tensor.int(), tensor)

        with pytest.raises(BaseError, match="floating-point"):
            reconstruction(tensor, tensor.int())

        with pytest.raises(ShapeError):
            reconstruction(tensor[0], tensor[0])

        with pytest.raises(ShapeError):
            reconstruction(tensor, tensor[..., :2])

        with pytest.raises(BaseError, match="method"):
            reconstruction(tensor, tensor, method="opening")

        with pytest.raises(BaseError, match="num_iters"):
            reconstruction(tensor, tensor, num_iters=-1)

        with pytest.raises(BaseError, match="check_every"):
            reconstruction(tensor, tensor, check_every=0)

        with pytest.raises(ValueCheckError, match="engine"):
            reconstruction(tensor, tensor, engine="convolution")

        with pytest.raises(BaseError, match="engine"):
            reconstruction(tensor, tensor, num_iters=0, engine="unknown")

        with pytest.raises(TypeCheckError):
            reconstruction(tensor, tensor, [[1.0]])

        with pytest.raises(ShapeError):
            reconstruction(tensor, tensor, torch.ones(3, 3, 3, device=device, dtype=dtype))

        with pytest.raises(BaseError, match="odd"):
            reconstruction(tensor, tensor, torch.ones(2, 3, device=device, dtype=dtype))

    def test_convolution_engine_rejected_with_checks_disabled(self, device, dtype):
        # The rejection guards termination, not input validity: on macOS CPU float32 the inexact `conv2d` step
        # makes the loop oscillate forever. It is not a KORNIA_CHECK, so disable_checks() leaves it on.
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        checks_were_enabled = are_checks_enabled()
        disable_checks()
        try:
            with pytest.raises(ValueCheckError, match="engine"):
                reconstruction(tensor, tensor, engine="convolution")
        finally:
            if checks_were_enabled:
                enable_checks()

    def test_jit(self, device, dtype):
        op_script = torch.jit.script(reconstruction)

        mask = torch.rand(1, 2, 7, 7, device=device, dtype=dtype)
        seed = mask * 0.5

        self.assert_close(op_script(seed, mask), reconstruction(seed, mask))

    @pytest.mark.parametrize("num_iters", [None, 3])
    def test_dynamo(self, device, dtype, torch_optimizer, num_iters):
        mask = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        seed = mask * 0.5
        op_optimized = torch_optimizer(reconstruction)

        self.assert_close(
            reconstruction(seed, mask, num_iters=num_iters), op_optimized(seed, mask, num_iters=num_iters)
        )

    @pytest.mark.parametrize("engine", ["unfold", "shift"])
    @pytest.mark.parametrize("method", ["dilation", "erosion"])
    def test_dynamo_fullgraph_num_iters(self, device, dtype, torch_optimizer, engine, method):
        # A fixed `num_iters` has no data-dependent exit, so the whole call compiles as one graph.
        mask = torch.rand(2, 3, 9, 9, device=device, dtype=dtype)
        seed = mask * 0.5 if method == "dilation" else mask + 0.5

        def op(s, m):
            return reconstruction(s, m, method=method, num_iters=3, engine=engine)

        self.assert_close(torch_optimizer(op, fullgraph=True)(seed, mask), op(seed, mask))
