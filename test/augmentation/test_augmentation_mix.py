import copy

import pytest
import torch

from kornia.augmentation import RandomCutMixV2, RandomJigsaw, RandomMixUpV2, RandomMosaic, RandomTransplantation
from kornia.testing import BaseTester, assert_close, tensor_to_gradcheck_var


class TestRandomMixUpV2:
    def test_smoke(self, device, dtype):
        f = RandomMixUpV2()
        repr = "RandomMixUpV2(lambda_val=None, p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr, str(f)

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.1320, 0.3074], device=device, dtype=dtype)

        expected = torch.stack(
            [
                torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
                torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1],
            ]
        )

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[:, 0], label)
        assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(p=0.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device)
        lam = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_lam0(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(lambda_val=(0.0, 0.0), p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.0, 0.0], device=device, dtype=dtype)

        expected = input.clone()

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[:, 0], label)
        assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(0)
        f = RandomMixUpV2(same_on_batch=True, p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)
        lam = torch.tensor([0.0885, 0.0885], device=device, dtype=dtype)

        expected = torch.stack(
            [
                torch.ones(1, 3, 4, device=device, dtype=dtype) * (1 - lam[0]),
                torch.ones(1, 3, 4, device=device, dtype=dtype) * lam[1],
            ]
        )

        out_image, out_label = f(input, label)
        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[:, 0], label)
        assert_close(out_label[:, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        assert_close(out_label[:, 2], lam, rtol=1e-4, atol=1e-4)


class TestRandomCutMixV2:
    def test_smoke(self):
        f = RandomCutMixV2(data_keys=["input", "class"])
        repr = "RandomCutMixV2(cut_size=None, beta=None, num_mix=1, p=1.0, p_batch=1.0, same_on_batch=False)"
        assert str(f) == repr

    def test_random_mixup_p1(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[0, :, 0], label)
        assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        assert_close(out_label[0, :, 2], torch.tensor([0.5, 0.5], device=device, dtype=dtype))

    def test_random_mixup_p0(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(p=0.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device)

        expected = input.clone()
        exp_label = torch.tensor([[[1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype)

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label, exp_label)

    def test_random_mixup_beta0(self, device, dtype):
        torch.manual_seed(76)
        # beta 0 => resample 0.5 area
        # beta cannot be 0 after torch 1.8.0
        f = RandomCutMixV2(beta=1e-7, p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[0, :, 0], label)
        assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        # cut area = 4 / 12
        assert_close(out_label[0, :, 2], torch.tensor([0.33333, 0.33333], device=device, dtype=dtype))

    def test_random_mixup_num2(self, device, dtype):
        torch.manual_seed(76)
        f = RandomCutMixV2(num_mix=5, p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[:, :, 0], label.view(1, -1).expand(5, 2))
        assert_close(
            out_label[:, :, 1], torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], device=device, dtype=dtype)
        )
        assert_close(
            out_label[:, :, 2],
            torch.tensor(
                [[0.0833, 0.3333], [0.0, 0.1667], [0.5, 0.0833], [0.0833, 0.0], [0.5, 0.3333]],
                device=device,
                dtype=dtype,
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_random_mixup_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        f = RandomCutMixV2(same_on_batch=True, p=1.0, data_keys=["input", "class"])

        input = torch.stack(
            [torch.ones(1, 3, 4, device=device, dtype=dtype), torch.zeros(1, 3, 4, device=device, dtype=dtype)]
        )
        label = torch.tensor([1, 0], device=device, dtype=dtype)

        expected = torch.tensor(
            [
                [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            ],
            device=device,
            dtype=dtype,
        )

        out_image, out_label = f(input, label)

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_label[0, :, 0], label)
        assert_close(out_label[0, :, 1], torch.tensor([0, 1], device=device, dtype=dtype))
        assert_close(
            out_label[0, :, 2], torch.tensor([0.5000, 0.5000], device=device, dtype=dtype), rtol=1e-4, atol=1e-4
        )


class TestRandomMosaic:
    def test_smoke(self):
        f = RandomMosaic(data_keys=["input", "class"])
        repr = (
            "RandomMosaic(output_size=None, mosaic_grid=(2, 2), start_ratio_range=(0.3, 0.7), p=0.7,"
            " p_batch=1.0, same_on_batch=False, mosaic_grid=(2, 2), output_size=None, min_bbox_size=0.0,"
            " padding_mode=constant, resample=bilinear, align_corners=True, cropping_mode=slice)"
        )
        assert str(f) == repr

    def test_numerical(self, device, dtype):
        torch.manual_seed(76)
        f = RandomMosaic(p=1.0, data_keys=["input", "bbox_xyxy"])

        input = torch.stack(
            [torch.ones(1, 8, 8, device=device, dtype=dtype), torch.zeros(1, 8, 8, device=device, dtype=dtype)]
        )
        boxes = torch.tensor([[[4, 5, 6, 7], [1, 2, 3, 4]], [[2, 2, 6, 6], [0, 0, 0, 0]]], device=device, dtype=dtype)

        out_image, out_box = f(input, boxes)

        expected = torch.tensor(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    ]
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )

        expected_box = torch.tensor(
            [
                [
                    [0.7074, 0.7099, 2.7074, 2.7099],
                    [0.0000, 0.0000, 1.0000, 1.0000],
                    [0.0000, 5.7099, 2.7074, 8.0000],
                    [0.0000, 2.7099, 1.0000, 4.7099],
                    [7.0000, 0.7099, 8.0000, 2.7099],
                    [5.7074, 0.0000, 7.7074, 1.0000],
                    [7.0000, 7.0000, 8.0000, 8.0000],
                    [5.7074, 5.7099, 7.7074, 7.7099],
                ],
                [
                    [0.0000, 0.0000, 1.0000, 2.8313],
                    [0.0000, 0.0000, 1.0000, 1.0000],
                    [0.0000, 7.0000, 1.0000, 8.0000],
                    [0.0000, 6.8313, 1.0000, 8.0000],
                    [4.5036, 0.0000, 8.0000, 2.8313],
                    [1.5036, 0.0000, 3.5036, 1.0000],
                    [4.5036, 6.8313, 8.0000, 8.0000],
                    [1.5036, 3.8313, 3.5036, 5.8313],
                ],
            ],
            device=device,
            dtype=dtype,
        )

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)
        assert_close(out_box, expected_box, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_p(self, p, device, dtype):
        torch.manual_seed(76)
        f = RandomMosaic(output_size=(300, 300), p=p, data_keys=["input", "bbox_xyxy"])

        input = torch.randn((2, 3, 224, 224), device=device, dtype=dtype)
        boxes = torch.tensor(
            [
                # image 1
                [[70.0, 5, 150, 100], [60, 180, 175, 220]],  # head  # feet
                # image 2
                [[75, 30, 175, 140], [0, 0, 0, 0]],  # head  # placeholder
            ],
            device=device,
            dtype=dtype,
        )

        f(input, boxes)


class TestRandomJigsaw:
    def test_smoke(self, device, dtype):
        f = RandomJigsaw(data_keys=["input"])
        repr = "RandomJigsaw(grid=(4, 4), p=0.5, p_batch=1.0, same_on_batch=False, grid=(4, 4))"
        assert str(f) == repr

        # Test square and non-square images.
        f = RandomJigsaw(grid=(2, 2), p=1.0, data_keys=["input"])
        input = torch.arange(64, device=device, dtype=dtype).reshape(2, 1, 4, 8)
        f(input)
        input = torch.arange(64, device=device, dtype=dtype).reshape(2, 1, 8, 4)
        f(input)
        input = torch.arange(32, device=device, dtype=dtype).reshape(2, 1, 4, 4)
        f(input)

    def test_numerical(self, device, dtype):
        torch.manual_seed(22)
        f = RandomJigsaw(grid=(2, 2), p=1.0, data_keys=["input"])

        input = torch.arange(32, device=device, dtype=dtype).reshape(2, 1, 4, 4)

        out_image = f(input)

        expected = torch.tensor(
            [
                [[[2.0, 3.0, 0.0, 1.0], [6.0, 7.0, 4.0, 5.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]],
                [
                    [
                        [16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0],
                        [24.0, 25.0, 26.0, 27.0],
                        [28.0, 29.0, 30.0, 31.0],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )

        assert_close(out_image, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_p(self, p, same_on_batch, device, dtype):
        torch.manual_seed(76)
        f = RandomJigsaw(p=p, data_keys=["input"], same_on_batch=same_on_batch)

        input = torch.randn((12, 3, 256, 256), device=device, dtype=dtype)

        f(input)


class TestRandomTransplantation(BaseTester):
    def test_smoke(self, device, dtype):
        torch.manual_seed(22)

        mask = torch.zeros(2, 3, 3, device=device, dtype=dtype)
        mask[0, 0:2, 0:2] = 1
        mask[1, 1:2, 1:2] = 2
        image = mask.clone().unsqueeze(dim=1)

        f = RandomTransplantation(p=1, excluded_labels=[0])
        image_out, mask_out = f(image, mask)

        mask_out_expected = torch.tensor(
            [[[1, 1, 0], [1, 2, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype
        )

        self.assert_close(mask_out, mask_out_expected)
        self.assert_close(image_out, mask_out_expected.unsqueeze(dim=1))

    def test_mask_only(self, device, dtype):
        torch.manual_seed(22)

        mask = torch.zeros(2, 3, 3, device=device, dtype=dtype)
        mask[0, 0:2, 0:2] = 1
        mask[1, 1:2, 1:2] = 2

        f = RandomTransplantation(p=1, excluded_labels=[0], data_keys=["mask"])
        mask_out = f(mask)

        mask_out_expected = torch.tensor(
            [[[1, 1, 0], [1, 2, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]], device=device, dtype=dtype
        )

        self.assert_close(mask_out, mask_out_expected)

    @pytest.mark.parametrize("n_spatial", [2, 3, 4])
    def test_module(self, n_spatial, device, dtype):
        torch.manual_seed(22)

        spatial_dimensions = [10] * n_spatial
        image = torch.rand(4, 3, *spatial_dimensions, device=device, dtype=dtype)
        mask = torch.zeros(4, *spatial_dimensions, device=device, dtype=dtype)
        mask_additional = torch.randint(0, 2, (4, *spatial_dimensions), device=device, dtype=dtype)

        selection = torch.zeros(*spatial_dimensions, device=device, dtype=torch.bool)
        selection[[slice(0, 5)] * n_spatial] = True
        assert selection.sum() == 5**n_spatial

        # Transplant rectangle from the (i - 1)-th to the i-th image
        for i in range(4):
            mask[i, selection] = i + 1

        image_copy = image.clone()
        mask_copy = mask.clone()
        mask_additional_copy = mask_additional.clone()

        f = RandomTransplantation(p=1, excluded_labels=[0])
        image_out, mask_out, mask_additional_out = f(image, mask, mask_additional, data_keys=["input", "mask", "mask"])

        self.assert_close(image, image_copy)
        self.assert_close(mask, mask_copy)
        self.assert_close(mask_additional, mask_additional_copy)

        for i in range(4):
            selection_moved = mask_out[i, selection]
            selection_unchanged = mask_out[i, ~selection]
            self.assert_close(selection_moved, torch.full_like(selection_moved, (i - 1) % 4 + 1))
            self.assert_close(selection_unchanged, torch.full_like(selection_unchanged, 0))
            self.assert_close(image_out[i, :, selection], image[(i - 1) % 4, :, selection])
            self.assert_close(image_out[i, :, ~selection], image[i, :, ~selection])
            self.assert_close(mask_additional_out[i, selection], mask_additional[(i - 1) % 4, selection])
            self.assert_close(mask_additional_out[i, ~selection], mask_additional[i, ~selection])

    def test_apply_none(self, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = RandomTransplantation(p=0)
        image_out, mask_out = f(image, mask)

        assert torch.all(f._params["batch_prob"] == 0)
        assert len(f._params["selected_labels"]) == 0

        self.assert_close(image_out, image)
        self.assert_close(mask_out, mask)

    def test_repeating(self, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = RandomTransplantation(p=0.5)
        image_out, mask_out = f(image, mask)
        image_out_same, mask_out_same = f(image, mask, params=f._params)
        image_out_different, mask_out_different = f(image, mask)

        self.assert_close(image_out, image_out_same)
        self.assert_close(mask_out, mask_out_same)
        with pytest.raises(AssertionError):
            self.assert_close(image_out, image_out_different)
        with pytest.raises(AssertionError):
            self.assert_close(mask_out, mask_out_different)

    def test_data_keys(self, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(4, 3, 10, 10, device=device, dtype=dtype)
        mask = torch.randint(0, 2, (4, 10, 10), device=device, dtype=dtype)

        f = RandomTransplantation(p=1)
        torch.manual_seed(22)
        image_out, mask_out = f(image, mask, data_keys=["input", "mask"])
        torch.manual_seed(22)
        mask_out2, image_out2 = f(mask, image, data_keys=["mask", "input"])

        self.assert_close(image_out, image_out2)
        self.assert_close(mask_out, mask_out2)

    @pytest.mark.parametrize(
        "input_shape_image, input_shape_mask, target_shape_image",
        [
            [(1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4)],  # (B, C, H, W)
            [(1, 2, 5, 3, 4), (1, 5, 3, 4), (1, 2, 5, 3, 4)],  # (B, C, D, H, W)
            [(1, 1, 1, 1), (1, 1, 1), (1, 1, 1, 1)],  # (B, C, H, W)
        ],
    )
    def test_cardinality(self, input_shape_image, input_shape_mask, target_shape_image, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(input_shape_image, device=device, dtype=dtype)
        mask = torch.randint(0, 2, input_shape_mask, device=device, dtype=dtype)

        f = RandomTransplantation(p=1)
        image_out, mask_out = f(image, mask)

        assert image_out.shape == target_shape_image
        assert mask_out.shape == torch.Size([s for i, s in enumerate(target_shape_image) if i != 1])

    def test_gradcheck(self, device):
        torch.manual_seed(22)
        image = torch.rand(1, 3, 2, 2, device=device, dtype=torch.float64)
        mask = torch.randint(0, 2, (1, 2, 2), device=device, dtype=torch.float64)

        image = tensor_to_gradcheck_var(image)  # to var
        mask = tensor_to_gradcheck_var(mask)  # to var

        assert self.gradcheck(RandomTransplantation(p=1.0), (image, mask), raise_exception=True, fast_mode=True)

    def test_exception(self, device, dtype):
        torch.manual_seed(22)
        image = torch.rand(1, 3, 2, 2, device=device, dtype=torch.float64)
        mask = torch.randint(0, 2, (1, 2, 2), device=device, dtype=torch.float64)
        f = RandomTransplantation(p=1.0)
        f(image, mask)
        params = f._params

        with pytest.raises(Exception, match="excluded_labels must be a 1-dimensional"):
            RandomTransplantation(p=1.0, excluded_labels=torch.tensor([[0, 1]], device=device, dtype=dtype))

        with pytest.raises(Exception, match="Length of keys.*does not match number of inputs"):
            f = RandomTransplantation(p=1.0)
            f(image, mask, data_keys=["input", "mask", "mask"])

        with pytest.raises(Exception, match="selected_labels must be a 1-dimensional tensor"):
            params_copy = copy.deepcopy(params)
            params_copy["selected_labels"] = torch.tensor([[0, 1]], device=device, dtype=dtype)
            f(image, mask, params=params_copy)

        with pytest.raises(Exception, match="Length of selected_labels.*does not match the number of images"):
            params_copy = copy.deepcopy(params)
            params_copy["selected_labels"] = torch.tensor([0, 1], device=device, dtype=dtype)
            f(image, mask, params=params_copy)

        with pytest.raises(Exception, match="Every image input must have one additional dimension"):
            f(image.unsqueeze(dim=-1), mask)

        with pytest.raises(Exception, match="The dimensions of the input image and segmentation mask must match"):
            image = torch.rand(1, 3, 2, 5, device=device, dtype=torch.float64)
            f(image, mask)
