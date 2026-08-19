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

import kornia

from testing.base import BaseTester


class TestRandomPerspective(BaseTester):
    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform_float(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)

        aug = kornia.augmentation.RandomPerspective(0.5, p=0.5)

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_smoke_no_transform(self, device, dtype):
        x_data = torch.rand(1, 2, 8, 9, dtype=dtype).to(device)

        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.5)

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_smoke_no_transform_batch(self, device, dtype):
        x_data = torch.rand(2, 2, 8, 9, dtype=dtype).to(device)

        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.5)

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_smoke_transform(self, device, dtype):
        x_data = torch.rand(1, 2, 4, 5, dtype=dtype).to(device)

        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.5)

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.transform_matrix.shape == torch.Size([1, 3, 3])
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_smoke_transform_sampling_method(self, device, dtype):
        x_data = torch.rand(1, 2, 4, 5, dtype=dtype).to(device)

        aug = kornia.augmentation.RandomPerspective(
            torch.tensor(0.5, device=device, dtype=dtype), p=0.5, sampling_method="area_preserving"
        )

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.transform_matrix.shape == torch.Size([1, 3, 3])
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_no_transform_module(self, device, dtype):
        x_data = torch.rand(1, 2, 8, 9, dtype=dtype).to(device)
        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype))
        out_perspective = aug(x_data)
        assert out_perspective.shape == x_data.shape
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_transform_module_should_return_identity(self, device, dtype):
        torch.manual_seed(0)
        x_data = torch.rand(1, 2, 4, 5, dtype=dtype).to(device)

        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.0)

        out_perspective = aug(x_data)
        assert out_perspective.shape == x_data.shape
        assert aug.transform_matrix.shape == (1, 3, 3)
        self.assert_close(out_perspective, x_data)
        self.assert_close(aug.transform_matrix, torch.eye(3, device=device, dtype=dtype)[None])
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_transform_module_should_return_expected_transform(self, device, dtype):
        torch.manual_seed(0)
        x_data = torch.rand(1, 2, 4, 5).to(device).type(dtype)

        # RandomPerspective samples with align_corners=False. expected_transform below is the
        # pixel-space transform and is unaffected by #3904; only the resampling changed, now that
        # warp_perspective normalizes to the convention grid_sample is actually called with.
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.1917, 0.3701, 0.1529, 0.0913, 0.2540],
                        [0.3200, 0.5976, 0.6278, 0.4746, 0.4695],
                        [0.0854, 0.4027, 0.2665, 0.1858, 0.1531],
                        [0.0000, 0.2196, 0.0264, 0.0000, 0.0000],
                    ],
                    [
                        [0.2632, 0.4507, 0.3209, 0.5829, 0.3465],
                        [0.3679, 0.7943, 0.4226, 0.2346, 0.2755],
                        [0.0746, 0.5083, 0.5565, 0.4573, 0.3209],
                        [0.0000, 0.0134, 0.0153, 0.0000, 0.0000],
                    ],
                ]
            ],
            device=device,
            dtype=x_data.dtype,
        )

        expected_transform = torch.tensor(
            [[[1.0523, 0.3493, 0.3046], [-0.1066, 1.0426, 0.5846], [0.0351, 0.1213, 1.0000]]],
            device=device,
            dtype=x_data.dtype,
        )

        aug = kornia.augmentation.RandomPerspective(
            torch.tensor(0.5, device=device, dtype=dtype), p=0.99999999
        )  # step one the random state

        out_perspective = aug(x_data)

        assert out_perspective.shape == x_data.shape
        assert aug.transform_matrix.shape == (1, 3, 3)
        self.assert_close(out_perspective, expected_output, atol=1e-4, rtol=1e-4)
        self.assert_close(aug.transform_matrix, expected_transform, atol=1e-4, rtol=1e-4)
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 5, 7, dtype=torch.float64, device=device)
        # TODO: turned off with p=0
        self.gradcheck(
            kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.0),
            (input,),
        )


class TestRandomAffine(BaseTester):
    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        aug = kornia.augmentation.RandomAffine(0.0)
        out = aug(x_data)
        assert out.shape == x_data.shape
        assert aug.inverse(out).shape == x_data.shape
        assert aug.inverse(out, aug._params).shape == x_data.shape

    def test_smoke_no_transform_batch(self, device):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        aug = kornia.augmentation.RandomAffine(0.0)
        out = aug(x_data)
        assert out.shape == x_data.shape
        # assert False, (aug.transform_matrix.shape, out.shape, aug._params)
        assert aug.inverse(out).shape == x_data.shape
        assert aug.inverse(out, aug._params).shape == x_data.shape

    @pytest.mark.parametrize("degrees", [45.0, (-45.0, 45.0), torch.tensor([45.0, 45.0])])
    @pytest.mark.parametrize("translate", [(0.1, 0.1), torch.tensor([0.1, 0.1])])
    @pytest.mark.parametrize(
        "scale", [(0.8, 1.2), (0.8, 1.2, 0.9, 1.1), torch.tensor([0.8, 1.2]), torch.tensor([0.8, 1.2, 0.7, 1.3])]
    )
    @pytest.mark.parametrize(
        "shear",
        [
            5.0,
            (-5.0, 5.0),
            (-5.0, 5.0, -3.0, 3.0),
            torch.tensor(5.0),
            torch.tensor([-5.0, 5.0]),
            torch.tensor([-5.0, 5.0, -3.0, 3.0]),
        ],
    )
    def test_batch_multi_params(self, degrees, translate, scale, shear, device, dtype):
        x_data = torch.rand(2, 2, 8, 9).to(device)
        aug = kornia.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        out = aug(x_data)
        assert out.shape == x_data.shape
        assert aug.inverse(out).shape == x_data.shape

    def test_smoke_transform(self, device):
        x_data = torch.rand(1, 2, 4, 5).to(device)
        aug = kornia.augmentation.RandomAffine(0.0)
        out = aug(x_data)

        assert out.shape == x_data.shape
        assert aug.transform_matrix.shape == torch.Size([1, 3, 3])
        assert aug.inverse(out).shape == x_data.shape

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7, device=device, dtype=torch.float64)
        # TODO: turned off with p=0
        self.gradcheck(kornia.augmentation.RandomAffine(10, p=0.0), (input,))


class TestRandomShear(BaseTester):
    torch.manual_seed(0)  # for random reproductibility

    def test_smoke_no_transform(self, device):
        x_data = torch.rand(1, 2, 8, 9).to(device)
        aug = kornia.augmentation.RandomShear((10.0, 10.0))
        out = aug(x_data)
        assert out.shape == x_data.shape
        assert aug.inverse(out).shape == x_data.shape
        assert aug.inverse(out, aug._params).shape == x_data.shape

    def test_gradcheck(self, device):
        input = torch.rand(1, 2, 5, 7, device=device, dtype=torch.float64)
        # TODO: turned off with p=0
        self.gradcheck(kornia.augmentation.RandomShear((10.0, 10.0), p=1.0), (input,))
