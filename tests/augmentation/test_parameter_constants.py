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

from functools import partial

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import kornia.augmentation as K
from kornia.augmentation import random_generator as rg
from kornia.augmentation.utils.helpers import _constant_tensor

from testing.base import BaseTester

# Numeric constructor inputs deliberately keep the generated parameters on CPU, except for the
# illumination generators, which follow the device and dtype their samplers were built for.
# Include both branches which construct destination corners in crop generators.
GENERATORS = [
    pytest.param(lambda: rg.AffineGenerator(30.0), (4, 3, 16, 19), id="affine"),
    pytest.param(lambda: rg.PerspectiveGenerator(0.5), (4, 3, 16, 19), id="perspective"),
    pytest.param(lambda: rg.ShearGenerator((-10.0, 10.0)), (4, 3, 16, 19), id="shear"),
    pytest.param(lambda: rg.RectangleEraseGenerator((0.1, 0.2), (0.8, 1.2), 0.5), (4, 3, 16, 19), id="erasing"),
    pytest.param(lambda: rg.CropGenerator((8, 9)), (4, 3, 16, 19), id="crop"),
    pytest.param(lambda: rg.CropGenerator((8, 9), resize_to=(6, 7)), (4, 3, 16, 19), id="crop-resize"),
    pytest.param(lambda: rg.ResizedCropGenerator((8, 9), (0.5, 1.0), (0.8, 1.2)), (4, 3, 16, 19), id="resized-crop"),
    pytest.param(lambda: partial(rg.center_crop_generator, size=(8, 9)), (4, 16, 19), id="center-crop"),
    pytest.param(lambda: rg.ResizeGenerator((8, 9)), (4, 3, 16, 19), id="resize"),
    pytest.param(lambda: rg.AffineGenerator3D(30.0), (4, 3, 12, 16, 19), id="affine3d"),
    pytest.param(lambda: rg.PerspectiveGenerator3D(0.5), (4, 3, 12, 16, 19), id="perspective3d"),
    pytest.param(lambda: rg.CropGenerator3D((6, 8, 9)), (4, 3, 12, 16, 19), id="crop3d"),
    pytest.param(lambda: rg.CropGenerator3D((6, 8, 9), resize_to=(4, 6, 7)), (4, 3, 12, 16, 19), id="crop3d-resize"),
    pytest.param(lambda: partial(rg.center_crop_generator3d, size=(6, 8, 9)), (4, 12, 16, 19), id="center-crop3d"),
    pytest.param(lambda: K.RandomThinPlateSpline(p=1.0).generate_parameters, (4, 3, 16, 19), id="tps"),
    pytest.param(lambda: rg.MosaicGenerator((8, 9)), (4, 3, 16, 19), id="mosaic"),
    pytest.param(rg.CutmixGenerator, (4, 3, 16, 19), id="cutmix"),
    pytest.param(lambda: rg.LinearIlluminationGenerator((0.1, 0.2), (0.1, 0.2)), (4, 3, 16, 19), id="linear"),
    pytest.param(
        lambda: rg.LinearCornerIlluminationGenerator((0.1, 0.2), (0.1, 0.2)), (4, 3, 16, 19), id="linear-corner"
    ),
    pytest.param(
        lambda: rg.GaussianIlluminationGenerator((0.1, 0.2), (0.1, 0.2), (0.1, 0.2), (0.1, 0.2)),
        (4, 3, 16, 19),
        id="gaussian",
    ),
]


def generate(generator, shape):
    return generator(*shape) if isinstance(generator, partial) else generator(shape)


@pytest.mark.parametrize("make_generator,shape", GENERATORS)
class TestParameterConstants(BaseTester):
    def test_no_lifted_constants(self, make_generator, shape):
        generator = make_generator()
        # Trace directly rather than through Dynamo: every tracer, not only torch.compile,
        # must see constants built by factories. Crop3D keeps data-dependent eager validation.
        graph = make_fx(lambda: generate(generator, shape), _error_on_data_dependent_ops=False)().graph
        lifted = [node for node in graph.nodes if node.target == torch.ops.aten.lift_fresh_copy.default]
        assert not lifted, f"Forward lifted fresh tensor constants: {lifted}"

    @pytest.mark.parametrize("moved", [False, True])
    def test_dynamo_transfer(self, make_generator, shape, moved, device, dtype, torch_optimizer):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Inductor regression covers CPU and CUDA")
        generator = make_generator()
        if moved and isinstance(generator, torch.nn.Module):
            generator.to(device=device, dtype=dtype)

        def transfer(input):
            params = generate(generator, shape)
            # Consume every parameter in an operation on the image device. Synchronizing
            # catches asynchronous invalid CPU-pointer accesses by generated CUDA kernels.
            return params, {name: value.to(input) + input for name, value in params.items()}

        input = torch.zeros((), device=device, dtype=dtype)
        compiled = torch_optimizer(transfer, fullgraph=not isinstance(generator, rg.CropGenerator3D))
        for _ in range(2):
            params, transferred = compiled(input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            for name, value in transferred.items():
                assert value.device == device
                assert value.dtype == dtype
                self.assert_close(value, params[name].to(input) + input)


AUGMENTATIONS = [
    pytest.param(lambda: K.RandomShear((-10.0, 10.0), p=1.0), (4, 3, 16, 19), id="shear"),
    pytest.param(lambda: K.RandomThinPlateSpline(p=1.0), (4, 3, 16, 19), id="tps"),
    pytest.param(lambda: K.RandomErasing(value=0.5, p=1.0), (4, 3, 16, 19), id="erasing"),
    pytest.param(lambda: K.CenterCrop((8, 9), cropping_mode="resample"), (4, 3, 16, 19), id="center-crop"),
    pytest.param(lambda: K.RandomCrop((8, 9), cropping_mode="resample"), (4, 3, 16, 19), id="crop"),
    pytest.param(lambda: K.RandomResizedCrop((8, 9), cropping_mode="resample"), (4, 3, 16, 19), id="resized-crop"),
    pytest.param(lambda: K.Resize((8, 9)), (4, 3, 16, 19), id="resize"),
    pytest.param(lambda: K.RandomAffine3D(30.0, p=1.0), (2, 3, 8, 10, 12), id="affine3d"),
    pytest.param(lambda: K.RandomPerspective3D(0.2, p=1.0), (2, 3, 8, 10, 12), id="perspective3d"),
    pytest.param(lambda: K.RandomCrop3D((4, 6, 7)), (2, 3, 8, 10, 12), id="crop3d"),
    pytest.param(lambda: K.CenterCrop3D((4, 6, 7)), (2, 3, 8, 10, 12), id="center-crop3d"),
]


class TestAugmentationConstantTransfer(BaseTester):
    @pytest.mark.parametrize("make_aug,shape", AUGMENTATIONS)
    @pytest.mark.parametrize("moved", [False, True])
    def test_dynamo_replay(self, make_aug, shape, moved, device, dtype, torch_optimizer):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Inductor regression covers CPU and CUDA")
        aug = make_aug()
        if moved:
            aug.to(device=device, dtype=dtype)
        input = torch.rand(shape, device=device, dtype=dtype)

        def apply(input):
            params = aug.forward_parameters(input.shape)
            return aug(input, params=params), params

        # RandomCrop reads replayable padding from tensors; Crop3D validates crop sizes.
        # These pre-existing graph breaks are separate from constant transfer safety.
        compiled = torch_optimizer(apply, fullgraph=not isinstance(aug, (K.RandomCrop, K.RandomCrop3D)))
        for _ in range(2):
            actual, params = compiled(input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.assert_close(actual, aug(input, params=params))


CONSTANT_DATA = [
    [],
    [[], []],
    [True, False],
    [257, 2049],
    [[0, 2048], [256, -1]],
    [[[1.5, -2.5]]],
    [0.0, -0.0, 1.0, 1],
    [[0.0, -0.0], [-0.0, 0.0]],
]


class TestConstantTensor(BaseTester):
    @pytest.mark.parametrize("data", CONSTANT_DATA)
    def test_values(self, data, device, dtype):
        expected = torch.tensor(data, device=device, dtype=dtype)
        actual = _constant_tensor(data, device=device, dtype=dtype)
        assert actual.shape == expected.shape
        assert actual.device == device
        assert actual.dtype == dtype
        self.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(actual.signbit(), expected.signbit())

    @pytest.mark.parametrize("data", CONSTANT_DATA)
    def test_dynamo_values(self, data, device, dtype, torch_optimizer):
        compiled = torch_optimizer(lambda: _constant_tensor(data, device=device, dtype=dtype), fullgraph=True)
        self.assert_close(compiled(), torch.tensor(data, device=device, dtype=dtype), rtol=0, atol=0)

    def test_integer_coordinates(self, device):
        data = [[4.5, 5.5], [12.5, 13.5]]
        expected = torch.tensor([[4, 5], [12, 13]], device=device, dtype=torch.long)
        self.assert_close(_constant_tensor(data, device=device, dtype=torch.long), expected)

    def test_dynamo_integer_coordinates(self, device, torch_optimizer):
        data = [[4.5, 5.5], [12.5, 13.5]]
        expected = torch.tensor([[4, 5], [12, 13]], device=device, dtype=torch.long)
        compiled = torch_optimizer(lambda: _constant_tensor(data, device=device, dtype=torch.long), fullgraph=True)
        self.assert_close(compiled(), expected)

    def test_numpy(self, device, dtype):
        data = np.array([[0.5, 1.0], [2.0, -3.0]])
        expected = torch.as_tensor(data, device=device, dtype=dtype)
        self.assert_close(_constant_tensor(data, device=device, dtype=dtype), expected, rtol=0, atol=0)

    def test_repeated_values_fill_once(self, device, dtype):
        # Box corners repeat their coordinates: one fill per distinct value, then one stack.
        data = [[[0, 0], [18, 0], [18, 15], [0, 15]]]
        graph = make_fx(lambda: _constant_tensor(data, device=device, dtype=dtype))().graph
        targets = [node.target for node in graph.nodes if node.op == "call_function"]
        assert targets.count(torch.ops.aten.full.default) == 3
        assert targets.count(torch.ops.aten.stack.default) == 1
        assert torch.ops.aten.lift_fresh_copy.default not in targets

    def test_signed_zeros_fill_once(self, device, dtype):
        # The fill key keeps the sign, so repeated zeros merge without turning -0.0 into 0.0.
        data = [[0.0, -0.0], [-0.0, 0.0], [0.0, 1.0]]
        graph = make_fx(lambda: _constant_tensor(data, device=device, dtype=dtype))().graph
        targets = [node.target for node in graph.nodes if node.op == "call_function"]
        assert targets.count(torch.ops.aten.full.default) == 3

    def test_dynamo_dynamic_size(self, device, dtype, torch_optimizer):
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Inductor regression covers CPU and CUDA")

        def dimensions(input):
            return (
                _constant_tensor([input.shape[-1] - 1, input.shape[-2] - 1], dtype=input.dtype).to(input) + input.sum()
            )

        compiled = torch_optimizer(dimensions, fullgraph=True, dynamic=True)
        for height, width in [(16, 19), (20, 23)]:
            input = torch.zeros(1, 1, height, width, device=device, dtype=dtype)
            self.assert_close(compiled(input), dimensions(input))


class TestShearSamplerPlacement(BaseTester):
    @pytest.mark.parametrize("make_generator", [rg.ShearGenerator, partial(rg.AffineGenerator, 30.0)])
    @pytest.mark.parametrize("shear", [10.0, (-10.0, 10.0), (-10.0, 10.0, -5.0, 5.0)])
    def test_move(self, make_generator, shear, device, dtype):
        generator = make_generator(shear=shear).to(device=device, dtype=dtype)
        for sampler in (generator.shear_x_sampler, generator.shear_y_sampler):
            assert sampler.low.device == device
            assert sampler.low.dtype == dtype
        generator((4, 3, 16, 19))


class TestIlluminationPlacement(BaseTester):
    @pytest.mark.parametrize(
        "make_generator",
        [
            pytest.param(lambda: rg.LinearIlluminationGenerator((0.1, 0.2), (-1.0, 1.0)), id="linear"),
            pytest.param(lambda: rg.LinearCornerIlluminationGenerator((0.1, 0.2), (-1.0, 1.0)), id="linear-corner"),
            pytest.param(
                lambda: rg.GaussianIlluminationGenerator((0.1, 0.2), (0.1, 0.2), (0.1, 0.2), (-1.0, 1.0)),
                id="gaussian",
            ),
        ],
    )
    def test_moved_sampler(self, make_generator, device, dtype):
        generator = make_generator().to(device=device, dtype=dtype)
        params = generator((4, 3, 16, 19))
        assert params["gradient"].device == device
        assert params["gradient"].dtype == dtype


class TestResizedCropNumpyScale(BaseTester):
    def test_matches_tuple(self, device, dtype):
        tuple_generator = rg.ResizedCropGenerator((8, 9), (0.5, 1.0), (0.8, 1.2)).to(device=device, dtype=dtype)
        numpy_generator = rg.ResizedCropGenerator((8, 9), np.array([0.5, 1.0]), (0.8, 1.2)).to(
            device=device, dtype=dtype
        )
        torch.manual_seed(0)
        expected = tuple_generator((4, 3, 16, 19))
        torch.manual_seed(0)
        actual = numpy_generator((4, 3, 16, 19))
        for name, value in expected.items():
            self.assert_close(actual[name], value, rtol=0, atol=0)

    def test_dynamo(self, device, dtype, torch_optimizer):
        generator = rg.ResizedCropGenerator((8, 9), np.array([0.5, 1.0]), (0.8, 1.2)).to(device=device, dtype=dtype)
        src = torch_optimizer(lambda: generator((4, 3, 16, 19))["src"], fullgraph=True)()
        assert src.shape == (4, 4, 2)
        assert bool(((src[..., 0] >= 0) & (src[..., 0] <= 18) & (src[..., 1] >= 0) & (src[..., 1] <= 15)).all())


class TestCenterCrop3DEmptyBatch(BaseTester):
    def test_matches_batched(self, device):
        empty = rg.center_crop_generator3d(0, 12, 16, 19, (6, 8, 9), device=device)
        batched = rg.center_crop_generator3d(1, 12, 16, 19, (6, 8, 9), device=device)
        for name, value in batched.items():
            assert empty[name].shape == (0, *value.shape[1:])
            assert empty[name].device == value.device
            assert empty[name].dtype == value.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Synchronization debugging requires CUDA")
class TestCudaParameterSync(BaseTester):
    @pytest.mark.parametrize(
        "make_generator",
        [
            pytest.param(lambda device: rg.AffineGenerator(torch.tensor([-30.0, 30.0], device=device)), id="affine"),
            pytest.param(lambda device: rg.PerspectiveGenerator(torch.tensor(0.5, device=device)), id="perspective"),
        ],
    )
    def test_no_host_device_sync(self, make_generator, device, dtype):
        if device.type != "cuda":
            pytest.skip("Synchronization debugging requires CUDA")
        generator = make_generator(device).to(device=device, dtype=dtype)
        generator((4, 3, 16, 19))
        torch.cuda.set_sync_debug_mode("error")
        try:
            generator((4, 3, 16, 19))
        finally:
            torch.cuda.set_sync_debug_mode("default")
