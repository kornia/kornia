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
from torch.fx.experimental.proxy_tensor import make_fx

import kornia
from kornia.core._compat import torch_version_lt

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


@pytest.mark.parametrize(
    "make_aug",
    [
        pytest.param(
            lambda tensor_range=False: kornia.augmentation.RandomAffine(
                degrees=torch.tensor(30.0, device="cpu") if tensor_range else 30.0,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                p=1.0,
            ),
            id="affine",
        ),
        pytest.param(
            lambda tensor_range=False: kornia.augmentation.RandomPerspective(
                torch.tensor(0.5, device="cpu") if tensor_range else 0.5, p=1.0
            ),
            id="perspective",
        ),
        pytest.param(
            lambda tensor_range=False: kornia.augmentation.RandomPerspective(
                torch.tensor(0.5, device="cpu") if tensor_range else 0.5, p=1.0, sampling_method="area_preserving"
            ),
            id="perspective-area-preserving",
        ),
    ],
)
class TestGeometricParameterDevice(BaseTester):
    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_geometry_constants_are_constructed_in_graph(self, make_aug):
        # CPU CI cannot execute the CUDA failure in #4516. Trace the deterministic geometry
        # and reject lifted tensor data, which Inductor may reuse across device transfers.
        generator = make_aug()._param_generator
        key = (
            "center" if isinstance(generator, kornia.augmentation.random_generator.AffineGenerator) else "start_points"
        )
        graphs = []

        def capture(module, inputs):
            graphs.append(make_fx(module)(*inputs).graph)
            return module.forward

        torch._dynamo.reset()
        torch.compile(lambda: generator((4, 3, 16, 19))[key].to(torch.float64), backend=capture, fullgraph=True)()
        assert len(graphs) == 1
        graph = graphs[0]
        # Also cover constants used for end_points and indexed scalar fills. Sampler
        # bounds may be captured as attributes, but forward must not lift fresh tensors.
        assert all(node.target != torch.ops.aten.lift_fresh_copy.default for node in graph.nodes)
        output = next(node for node in graph.nodes if node.op == "output")
        pending = list(output.all_input_nodes)
        visited = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            assert node.op != "get_attr", f"Lifted geometry constant: {node.target}"
            pending.extend(node.all_input_nodes)

    @pytest.mark.parametrize("batch_size", [0, 1, 4])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_numeric_parameters_keep_default_placement(self, make_aug, batch_size, same_on_batch, device, dtype):
        aug = make_aug().to(device=device, dtype=dtype)
        aug.same_on_batch = same_on_batch
        params = aug.forward_parameters((batch_size, 3, 8, 9))
        for name, value in params.items():
            if name in ("batch_prob", "forward_input_shape"):
                continue
            assert value.device == torch.device("cpu"), name
            # Returned precision remains separate from the sampler's requested precision.
            assert value.dtype == torch.get_default_dtype(), name
            if same_on_batch and batch_size > 1 and name not in ("center", "start_points"):
                self.assert_close(value, value[:1].expand_as(value))

    def test_container_move(self, make_aug, device, dtype):
        aug = make_aug()
        sequence = kornia.augmentation.AugmentationSequential(aug, data_keys=["input"]).to(device=device, dtype=dtype)
        input = torch.rand(2, 3, 8, 9, device=device, dtype=dtype)
        output = sequence(input)
        assert output.device == device
        assert output.dtype == dtype
        for name, value in aug._params.items():
            if name not in ("batch_prob", "forward_input_shape", "data_keys"):
                assert value.device == torch.device("cpu"), name

    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_default_dtype_after_sampler_move(self, make_aug, device, default_dtype):
        # A float64 default must not force float64 sampling on MPS.
        aug = make_aug().to(device=device, dtype=torch.float32)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            params = aug.forward_parameters((4, 3, 8, 9))
            generator = aug._param_generator
            assert generator.dtype == torch.float32
            sampler = generator.degree_sampler if hasattr(generator, "degree_sampler") else generator.rand_val_sampler
            assert sampler.low.dtype == torch.float32
            assert sampler.low.device == device
            for name, value in params.items():
                if name not in ("batch_prob", "forward_input_shape"):
                    assert value.device == torch.device("cpu"), name
                    assert value.dtype == default_dtype, name
        finally:
            torch.set_default_dtype(original_dtype)

    def test_ambient_default_device(self, make_aug, device, dtype):
        # Sampling stays on CPU while numeric returned parameters follow the default device.
        original_modes = torch.overrides._get_current_function_mode_stack()
        with torch.device(device):
            aug = make_aug()
            params = aug.forward_parameters((4, 3, 8, 9))
            assert aug._param_generator.device == torch.device("cpu")
            for name, value in params.items():
                if name not in ("batch_prob", "forward_input_shape"):
                    assert value.device == device, name
        assert torch.overrides._get_current_function_mode_stack() == original_modes

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize(
        "placement",
        ["unmoved", "moved", "cpu_tensor", "cpu_tensor_unmoved", "default_device", "dtype_only", "container"],
    )
    def test_dynamo_parameter_generation(self, make_aug, batch_size, placement, device, dtype, torch_optimizer):
        # #4516: capture fresh parameter generation and CPU-to-image transfers in one graph.
        # Replaying precomputed parameters alone does not reproduce the Inductor defect.
        if device.type not in ("cpu", "cuda"):
            pytest.skip("Inductor regression covers CPU and CUDA")
        if placement == "container" and torch_version_lt(2, 6, 0):
            pytest.skip("PyTorch 2.5 cannot trace the container's isinstance checks with union types")
        with torch.device(device if placement == "default_device" else "cpu"):
            aug = make_aug(tensor_range=placement in ("cpu_tensor", "cpu_tensor_unmoved"))
            if placement in ("moved", "cpu_tensor"):
                aug.to(device=device, dtype=dtype)
            elif placement == "dtype_only":
                aug.to(device).to(dtype)
            module = aug
            if placement == "container":
                module = kornia.augmentation.AugmentationSequential(aug, data_keys=["input"]).to(device, dtype)
                # Fullgraph captures the tensor pipeline; PIL/NumPy conversion uses Python decorators.
                module.disable_features = True
            input = torch.rand(batch_size, 3, 16, 19, device=device, dtype=dtype)

            def apply_with_parameters(input):
                # Capture sampling with application, and return parameters explicitly:
                # older Torch export guards suppress the augmentation's _params side effect.
                params = module.forward_parameters(input.shape)
                output = module(input, params=params)
                return output, params[0].data if placement == "container" else params

            compiled = torch_optimizer(apply_with_parameters, fullgraph=True)
            for _ in range(2):
                actual, params = compiled(input)
                expected = aug(input, params=params)
                self.assert_close(actual, expected)
                parameter_device = device if placement == "default_device" else torch.device("cpu")
                for name, value in params.items():
                    if name not in ("batch_prob", "forward_input_shape", "data_keys"):
                        assert value.device == parameter_device, name


class TestGeometricTensorRangeDevice(BaseTester):
    def test_perspective_tensor_range_keeps_placement(self, device, dtype):
        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device="cpu", dtype=dtype), p=1.0)
        aug.set_rng_device_and_dtype(device, torch.float32)
        params = aug.forward_parameters((4, 3, 8, 9))
        for name in ("start_points", "end_points"):
            assert params[name].device == torch.device("cpu")
            assert params[name].dtype == dtype
