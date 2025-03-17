import pytest
import torch

import kornia
from kornia.augmentation import RandomCrop, RandomCrop3D
from testing.base import BaseTester


class TestRandomCrop3D(BaseTester):
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop3D(size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, p=1.0)
        repr = (
            "RandomCrop3D(crop_size=(2, 3, 4), padding=(0, 1, 2), fill=10, pad_if_needed=False, "
            "padding_mode=constant, resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, "
            "return_transform=None)"
        )
        assert str(f) == repr

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_no_padding(self, batch_size, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor(
            [
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10, 11, 12, 13, 14],
                            [15, 16, 17, 18, 19],
                            [20, 21, 22, 23, 24],
                        ]
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(batch_size, 1, 5, 1, 1)
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1.0)
        out = f(input_tensor)
        if batch_size == 1:
            expected = torch.tensor(
                [[[[[11, 12, 13, 14], [16, 17, 18, 19], [21, 22, 23, 24]]]]], device=device, dtype=dtype
            ).repeat(batch_size, 1, 2, 1, 1)
        if batch_size == 2:
            expected = torch.tensor(
                [
                    [
                        [
                            [
                                [6.0000, 7.0000, 8.0000, 9.0000],
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                            ],
                            [
                                [6.0000, 7.0000, 8.0000, 9.0000],
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                            ],
                        ]
                    ],
                    [
                        [
                            [
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                                [21.0000, 22.0000, 23.0000, 24.0000],
                            ],
                            [
                                [11.0000, 12.0000, 13.0000, 14.0000],
                                [16.0000, 17.0000, 18.0000, 19.0000],
                                [21.0000, 22.0000, 23.0000, 24.0000],
                            ],
                        ]
                    ],
                ],
                device=device,
                dtype=dtype,
            )

        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop3D(size=(2, 3, 4), padding=None, align_corners=True, p=1.0, same_on_batch=True)
        input_tensor = (
            torch.eye(6, device=device, dtype=dtype)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .repeat(2, 3, 5, 1, 1)
        )
        res = f(input_tensor)
        self.assert_close(res[0], res[1])

    @pytest.mark.parametrize("padding", [1, (1, 1, 1), (1, 1, 1, 1, 1, 1)])
    def test_padding_batch(self, padding, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        input_tensor = torch.tensor(
            [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]], device=device, dtype=dtype
        ).repeat(batch_size, 1, 3, 1, 1)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.0, 1.0, 2.0, 10.0], [3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0]],
                        [[0.0, 1.0, 2.0, 10.0], [3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0]],
                    ]
                ],
                [
                    [
                        [[3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0], [10, 10, 10, 10.0]],
                        [[3.0, 4.0, 5.0, 10.0], [6.0, 7.0, 8.0, 10.0], [10, 10, 10, 10.0]],
                    ]
                ],
            ],
            device=device,
            dtype=dtype,
        )
        f = RandomCrop3D(size=(2, 3, 4), fill=10.0, padding=padding, align_corners=True, p=1.0)
        out = f(input_tensor)

        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor([[[0.0, 1.0, 2.0]]], device=device, dtype=dtype)
        expected = torch.tensor(
            [
                [
                    [
                        [[9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]],
                        [[0.0, 1.0, 2.0, 9.0], [9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0]],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        rc = RandomCrop3D(size=(2, 3, 4), pad_if_needed=True, fill=9, align_corners=True, p=1.0)
        out = rc(input_tensor)

        self.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility
        input_tensor = torch.rand((3, 3, 3), device=device, dtype=torch.float64)  # 3 x 3
        self.gradcheck(RandomCrop3D(size=(3, 3, 3), p=1.0), (input_tensor,))

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.0).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.geometry.transform.center_crop3d(img)
        self.assert_close(actual, expected)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit_trace(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.0).forward
        op_script = torch.jit.script(op)
        # 1. Trace op
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        op_trace = torch.jit.trace(op_script, (img,))

        # 2. Generate new input
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        # 3. Evaluate
        actual = op_trace(img)
        expected = op(img)
        self.assert_close(actual, expected)