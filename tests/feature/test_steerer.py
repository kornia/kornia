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

from kornia.feature.steerers import DiscreteSteerer

from testing.base import BaseTester

class TestDiscreteSteerer(BaseTester):

    @pytest.mark.parametrize("num_desc, desc_dim, steerer_power", [(1, 4, 1), (2, 128, 7), (32, 128, 11)])
    def test_shape(self, num_desc, desc_dim, steerer_power, device):
        desc = torch.rand(num_desc, desc_dim, device=device)
        generator = torch.rand(desc_dim, desc_dim, device=device)
        steerer = DiscreteSteerer(generator)
        desc = steerer.steer_descriptions(desc, steerer_power=steerer_power)
        assert desc.shape == (num_desc, desc_dim)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_steering(self, device, normalize):
        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        desc = torch.rand(16, 2, device=device)
        steerer = DiscreteSteerer(generator)
        desc_out = steerer.steer_descriptions(desc, steerer_power=3, normalize=normalize)
        if normalize:
            desc = torch.nn.functional.normalize(desc, dim=-1)
        desc = desc[:, [1, 0]]
        desc[:, 0] = -desc[:, 0]
        assert torch.allclose(desc, desc_out, atol=1e-6)

    @pytest.mark.parametrize("generator_type", ["C4", "SO2"])
    @pytest.mark.parametrize("steerer_order", [2, 14])
    def test_default(self, device, generator_type, steerer_order):
        steerer = DiscreteSteerer.create_dedode_default(
            generator_type=generator_type,
            steerer_order=steerer_order,
        ).to(device)
        assert isinstance(steerer, DiscreteSteerer)
        shape = (96, 256)
        desc = torch.randn(*shape, device=device)
        desc = steerer(desc)
        assert desc.shape == shape

    @pytest.mark.parametrize("steerer_power", [0, 1, 5, -1])
    def test_steerer_power_extremes(self, device, steerer_power):
        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        desc = torch.rand(8, 2, device=device)
        steerer = DiscreteSteerer(generator)
        out = steerer.steer_descriptions(desc, steerer_power=steerer_power)
        assert out.shape == desc.shape

    def test_normalization_preserves_norm(self, device):
        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        desc = torch.rand(8, 2, device=device)
        steerer = DiscreteSteerer(generator)
        out = steerer.steer_descriptions(desc, steerer_power=1, normalize=True)
        orig_norm = torch.norm(out, dim=-1)
        assert torch.allclose(orig_norm, torch.ones_like(orig_norm), atol=1e-6)

    def test_invalid_generator_type_raises(self):
        with pytest.raises(ValueError):
            DiscreteSteerer.create_dedode_default(generator_type="INVALID")

    def test_gradient_flow(self, device):
        generator = torch.tensor([[0.0, 1], [-1, 0]], device=device)
        desc = torch.rand(4, 2, device=device, requires_grad=True)
        steerer = DiscreteSteerer(generator)
        out = steerer.steer_descriptions(desc, steerer_power=2)
        loss = out.sum()
        loss.backward()
        assert desc.grad is not None
        assert torch.all(desc.grad != 0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_vs_cuda_consistency(self):
        generator = torch.tensor([[0.0, 1], [-1, 0]])
        desc = torch.rand(4, 2)
        cpu_steerer = DiscreteSteerer(generator)
        cuda_steerer = DiscreteSteerer(generator.to("cuda"))
        out_cpu = cpu_steerer.steer_descriptions(desc, steerer_power=2)
        out_cuda = cuda_steerer.steer_descriptions(desc.to("cuda"), steerer_power=2).cpu()
        assert torch.allclose(out_cpu, out_cuda, atol=1e-6)

