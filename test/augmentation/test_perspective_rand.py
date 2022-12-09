import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.testing import assert_close


class TestRandomPerspective:

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
        assert_close(out_perspective, x_data)
        assert_close(aug.transform_matrix, torch.eye(3, device=device, dtype=dtype)[None])
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
        assert_close(out_perspective, expected_output, atol=1e-4, rtol=1e-4)
        assert_close(aug.transform_matrix, expected_transform, atol=1e-4, rtol=1e-4)
        assert aug.inverse(out_perspective).shape == x_data.shape

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 5, 7, dtype=dtype).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        # TODO: turned off with p=0
        assert gradcheck(
            kornia.augmentation.RandomPerspective(torch.tensor(0.5, device=device, dtype=dtype), p=0.0),
            (input,),
            raise_exception=True,
            fast_mode=True,
        )


class TestRandomAffine:

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
        input = torch.rand(1, 2, 5, 7).to(device)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        # TODO: turned off with p=0
        assert gradcheck(kornia.augmentation.RandomAffine(10, p=0.0), (input,), raise_exception=True, fast_mode=True)
