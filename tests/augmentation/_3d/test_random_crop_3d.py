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
from kornia.augmentation import RandomCrop, RandomCrop3D

from testing.base import BaseTester


class TestRandomCrop3D(BaseTester):
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        "padding,pad_if_needed,size",
        [
            (None, False, (2, 3, 4)),
            (2, False, (2, 3, 4)),
            ((1, 2, 3), False, (2, 3, 4)),
            ((1, 2, 3, 4, 1, 2), False, (2, 3, 4)),
            (None, True, (6, 7, 9)),
            (1, True, (8, 9, 10)),
        ],
    )
    def test_skipped_padding(self, batch_size, padding, pad_if_needed, size, device, dtype):
        input_tensor = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6)
        input_tensor = input_tensor.repeat(batch_size, 1, 1, 1, 1).requires_grad_()
        aug = RandomCrop3D(size, padding=padding, pad_if_needed=pad_if_needed, p=0.0)

        output = aug(input_tensor)

        self.assert_close(output, input_tensor, rtol=0, atol=0)
        self.assert_close(aug.transform_matrix, torch.eye(4, device=device, dtype=dtype).expand(batch_size, 4, 4))
        self.assert_close(aug(input_tensor, params=aug._params), input_tensor, rtol=0, atol=0)
        output.sum().backward()
        self.assert_close(input_tensor.grad, torch.ones_like(input_tensor), rtol=0, atol=0)

    @pytest.mark.parametrize("shape", [(4, 5, 6), (2, 4, 5, 6), (2, 1, 4, 5, 6)])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_skipped_padding_keepdim(self, shape, keepdim, device, dtype):
        input_tensor = torch.ones(shape, device=device, dtype=dtype)
        aug = RandomCrop3D((6, 7, 9), padding=1, pad_if_needed=True, p=0.0, keepdim=keepdim)
        expected = input_tensor if keepdim else input_tensor.reshape((1,) * (5 - len(shape)) + shape)

        self.assert_close(aug(input_tensor), expected, rtol=0, atol=0)

    def test_skipped_padding_params(self, device, dtype):
        input_tensor = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6)
        aug = RandomCrop3D((2, 3, 4), padding=2, p=0.5)
        # Use explicit decisions so the test does not depend on random draws.
        params = aug.forward_parameters(input_tensor.shape)
        params["batch_prob"] = torch.ones_like(params["batch_prob"])
        assert aug(input_tensor, params=params).shape == (1, 1, 2, 3, 4)
        params = {**params, "batch_prob": torch.zeros_like(params["batch_prob"])}

        self.assert_close(aug(input_tensor, params=params), input_tensor, rtol=0, atol=0)
        self.assert_close(aug.transform_matrix, torch.eye(4, device=device, dtype=dtype).unsqueeze(0))

    @pytest.mark.parametrize("padding_mode", ["constant", "replicate", "reflect"])
    @pytest.mark.parametrize("padding", [None, 1, (1, 2, 1), (1, 2, 1, 2, 1, 2)])
    def test_padding_replay(self, padding, padding_mode, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.arange(120, device=device, dtype=dtype).reshape(1, 1, 4, 5, 6).repeat(2, 1, 1, 1, 1)
        input_tensor = input_tensor / 120
        # Nearest sampling makes integer-coordinate crops match slicing, even in half precision.
        aug = RandomCrop3D(
            (7, 8, 9), padding=padding, pad_if_needed=True, padding_mode=padding_mode, resample="nearest", p=1.0
        )
        padded = aug.precrop_padding(input_tensor)
        output = aug(input_tensor)

        assert output.shape == (2, 1, 7, 8, 9)
        for index, box in enumerate(aug._params["src"]):
            x, y, z = box[0].long().tolist()
            self.assert_close(output[index], padded[index, :, z : z + 7, y : y + 8, x : x + 9], rtol=0, atol=0)
        self.assert_close(aug(input_tensor, params=aug._params), output, rtol=0, atol=0)

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

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_rejects_a_crop_one_voxel_larger_than_the_input(self, axis, device, dtype):
        # The size guard counted valid start offsets and rejected only a negative count, so a crop exactly
        # one voxel too large passed and the output gained an empty slab.
        volume = torch.ones(1, 1, 4, 5, 6, device=device, dtype=dtype)
        size = [4, 5, 6]
        size[axis] += 1
        with pytest.raises(ValueError, match="cannot be smaller than crop size"):
            RandomCrop3D(tuple(size), p=1.0)(volume)
        # The whole volume is still a valid crop.
        assert RandomCrop3D((4, 5, 6), p=1.0)(volume).shape == (1, 1, 4, 5, 6)
        # Padding is counted in, so the same size fits once the axis is padded.
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - axis)] = 1
        assert RandomCrop3D(tuple(size), padding=tuple(padding), p=1.0)(volume).shape == (1, 1, *size)

    def test_fill_accepts_one_value_per_channel(self, device, dtype):
        volume = torch.zeros(1, 3, 2, 2, 2, device=device, dtype=dtype)
        fill = (0.25, 0.5, 0.75)
        padded = RandomCrop3D((4, 4, 4), padding=1, fill=fill, p=1.0).precrop_padding(volume)
        assert padded.shape == (1, 3, 4, 4, 4)
        expected = volume.new_tensor(fill).view(1, 3, 1, 1, 1)
        self.assert_close(padded[:, :, 0, 0, 0], expected[:, :, 0, 0, 0])
        self.assert_close(padded[:, :, -1, -1, -1], expected[:, :, 0, 0, 0])
        self.assert_close(padded[:, :, 1:3, 1:3, 1:3], volume)  # the interior is untouched
        # A scalar keeps the old behaviour.
        scalar = RandomCrop3D((4, 4, 4), padding=1, fill=7.0, p=1.0).precrop_padding(volume)
        self.assert_close(scalar[:, :, 0, 0, 0], torch.full((1, 3), 7.0, device=device, dtype=dtype))

    @pytest.mark.device_agnostic
    def test_fill_sequence_is_validated(self):
        volume = torch.zeros(1, 3, 2, 2, 2)
        with pytest.raises(ValueError, match="one value per channel"):
            RandomCrop3D((4, 4, 4), padding=1, fill=(1.0, 0.0), p=1.0).precrop_padding(volume)
        with pytest.raises(ValueError, match="padding_mode='constant'"):
            RandomCrop3D((4, 4, 4), padding=1, fill=(1.0, 0.0, 0.0), padding_mode="replicate", p=1.0).precrop_padding(
                volume
            )
