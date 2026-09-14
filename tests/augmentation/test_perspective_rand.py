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


@pytest.mark.parametrize(
    "make_aug",
    [
        pytest.param(
            lambda: kornia.augmentation.RandomAffine(degrees=30.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
            id="affine",
        ),
        pytest.param(lambda: kornia.augmentation.RandomPerspective(0.5, p=1.0), id="perspective"),
        pytest.param(
            lambda: kornia.augmentation.RandomPerspective(0.5, p=1.0, sampling_method="area_preserving"),
            id="perspective-area-preserving",
        ),
    ],
)
class TestGeometricParameterDevice(BaseTester):
    @pytest.mark.parametrize("batch_size", [0, 1, 4])
    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_numeric_parameters_follow_rng_device(self, make_aug, batch_size, same_on_batch, device, dtype):
        aug = make_aug().to(device=device, dtype=dtype)
        aug.same_on_batch = same_on_batch
        params = aug.forward_parameters((batch_size, 3, 8, 9))
        for name, value in params.items():
            if name in ("batch_prob", "forward_input_shape"):
                continue
            assert value.device == device, name
            # Returned precision remains separate from the sampler's requested precision.
            assert value.dtype == torch.get_default_dtype(), name
            if same_on_batch and batch_size:
                self.assert_close(value, value[:1].expand_as(value))

    def test_container_move(self, make_aug, device, dtype):
        aug = make_aug()
        sequence = kornia.augmentation.AugmentationSequential(aug, data_keys=["input"]).to(device=device, dtype=dtype)
        input = torch.rand(2, 3, 8, 9, device=device, dtype=dtype)
        output = sequence(input)
        assert output.device == device
        assert output.dtype == dtype
        for name, value in aug._params.items():
            if name not in ("forward_input_shape", "data_keys"):
                assert value.device == device, name

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_dynamo_numeric_parameters(self, make_aug, batch_size, device, dtype, torch_optimizer):
        # #4516: compiling only apply_transform with precomputed parameters misses the
        # CPU constants passed to CUDA kernels by the full parameter-generation graph.
        aug = make_aug().to(device=device, dtype=dtype)
        input = torch.rand(batch_size, 3, 16, 19, device=device, dtype=dtype)
        compiled = torch_optimizer(aug)
        for _ in range(2):
            actual = compiled(input)
            expected = aug(input, params=aug._params)
            self.assert_close(actual, expected)


class TestGeometricTensorRangeDevice(BaseTester):
    @pytest.mark.parametrize("range_name", ["degrees", "translate", "scale", "shear"])
    def test_affine_tensor_range_keeps_placement(self, range_name, device, dtype):
        ranges = {"degrees": 30.0, "translate": (0.1, 0.1), "scale": (0.8, 1.2), "shear": (0.0, 5.0, 0.0, 5.0)}
        # Any tensor-valued range, including an optional one, controls returned placement.
        ranges[range_name] = torch.tensor(ranges[range_name], device=device, dtype=dtype)
        aug = kornia.augmentation.RandomAffine(**ranges, p=1.0)
        aug.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
        params = aug.forward_parameters((4, 3, 8, 9))
        for name, value in params.items():
            if name not in ("batch_prob", "forward_input_shape"):
                assert value.device == device, name
                assert value.dtype == dtype, name

    def test_perspective_tensor_range_keeps_placement(self, device, dtype):
        aug = kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=1.0)
        aug.set_rng_device_and_dtype(torch.device("cpu"), torch.float32)
        params = aug.forward_parameters((4, 3, 8, 9))
        for name in ("start_points", "end_points"):
            assert params[name].device == device
            assert params[name].dtype == dtype


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

        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0197, 0.0429],
                        [0.0000, 0.5632, 0.5322, 0.3677, 0.1430],
                        [0.0000, 0.3083, 0.4032, 0.1761, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000, 0.1189, 0.0586],
                        [0.0000, 0.7087, 0.5420, 0.3995, 0.0863],
                        [0.0000, 0.2695, 0.5981, 0.5888, 0.0000],
                        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
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
