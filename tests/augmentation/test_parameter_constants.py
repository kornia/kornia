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

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import kornia.augmentation as K
from kornia.augmentation import random_generator as rg
from kornia.augmentation.utils.helpers import _constant_tensor

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available

# Numeric constructor inputs deliberately keep the generated parameters on CPU.
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
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_no_lifted_constants(self, make_generator, shape):
        generator = make_generator()
        graphs = []

        def capture(module, inputs):
            graphs.append(make_fx(module, _error_on_data_dependent_ops=False)(*inputs).graph)
            return module.forward

        torch._dynamo.reset()
        # Crop3D retains data-dependent eager validation and may split the graph.
        torch.compile(lambda: generate(generator, shape), backend=capture, fullgraph=False)()
        assert graphs
        nodes = [node for graph in graphs for node in graph.nodes]
        lifted = [node for node in nodes if node.target == torch.ops.aten.lift_fresh_copy.default]
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


class TestConstantTensor(BaseTester):
    @pytest.mark.parametrize("data", [[], [True, False], [257, 2049], [[0, 2048], [256, -1]], [[[1.5, -2.5]]]])
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_values(self, data, device, dtype):
        torch._dynamo.reset()
        expected = torch.tensor(data, device=device, dtype=dtype)
        actual = _constant_tensor(data, device=device, dtype=dtype)
        self.assert_close(actual, expected, rtol=0, atol=0)
        compiled = torch.compile(
            lambda: _constant_tensor(data, device=device, dtype=dtype), backend="eager", fullgraph=True
        )
        self.assert_close(compiled(), expected, rtol=0, atol=0)

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_integer_coordinates(self, device):
        torch._dynamo.reset()
        data = [[4.5, 5.5], [12.5, 13.5]]
        expected = torch.tensor([[4, 5], [12, 13]], device=device, dtype=torch.long)
        compiled = torch.compile(
            lambda: _constant_tensor(data, device=device, dtype=torch.long), backend="eager", fullgraph=True
        )
        self.assert_close(compiled(), expected)

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
    @pytest.mark.parametrize("shear", [10.0, (-10.0, 10.0), (-10.0, 10.0, -5.0, 5.0)])
    def test_move(self, shear, device, dtype):
        generator = rg.ShearGenerator(shear).to(device=device, dtype=dtype)
        for sampler in (generator.shear_x_sampler, generator.shear_y_sampler):
            assert sampler.low.device == device
            assert sampler.low.dtype == dtype
        generator((4, 3, 16, 19))


class TestLinearIlluminationPlacement(BaseTester):
    @pytest.mark.parametrize("generator_type", [rg.LinearIlluminationGenerator, rg.LinearCornerIlluminationGenerator])
    def test_moved_sampler(self, generator_type, device, dtype):
        generator = generator_type((0.1, 0.2), (-1.0, 1.0)).to(device=device, dtype=dtype)
        params = generator((4, 3, 16, 19))
        assert params["gradient"].device == torch.device("cpu")
        assert params["gradient"].dtype == torch.get_default_dtype()
