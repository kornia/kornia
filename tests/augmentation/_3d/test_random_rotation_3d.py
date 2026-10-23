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

from kornia.augmentation import RandomRotation3D
from kornia.augmentation.container.augment import AugmentationSequential

from testing.base import BaseTester


class TestRandomRotation3D(BaseTester):
    torch.manual_seed(0)  # for random reproductibility

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation3D(degrees=45.5)
        repr = (
            """RandomRotation3D(degrees=tensor([[-45.5000, 45.5000],
        [-45.5000, 45.5000],
        [-45.5000, 45.5000]]), resample=BILINEAR, align_corners=False, p=0.5, """
            """p_batch=1.0, same_on_batch=False, return_transform=None)"""
        )
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation3D(degrees=45.0)

        input_tensor = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )  # 3 x 4 x 4

        expected = torch.tensor(
            [
                [
                    [
                        [
                            [0.0000, 0.0000, 0.6810, 0.5250],
                            [0.5052, 0.0000, 0.0000, 0.0613],
                            [0.1159, 0.1072, 0.5324, 0.0870],
                            [0.0000, 0.0000, 0.1927, 0.0000],
                        ],
                        [
                            [0.0000, 0.1683, 0.6963, 0.1131],
                            [0.0566, 0.0000, 0.5215, 0.2796],
                            [0.0694, 0.6039, 1.4519, 1.1240],
                            [0.0000, 0.1325, 0.1542, 0.2510],
                        ],
                        [
                            [0.0000, 0.2054, 0.0000, 0.0000],
                            [0.0026, 0.6088, 0.7358, 0.2319],
                            [0.1261, 1.0830, 1.3687, 1.4940],
                            [0.0000, 0.0416, 0.2012, 0.3124],
                        ],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )

        expected_transform = torch.tensor(
            [
                [
                    [0.6523, 0.3666, -0.6635, 0.6352],
                    [-0.6185, 0.7634, -0.1862, 1.4689],
                    [0.4382, 0.5318, 0.7247, -1.1797],
                    [0.0000, 0.0000, 0.0000, 1.0000],
                ]
            ],
            device=device,
            dtype=dtype,
        )

        out = f(input_tensor)
        atol = 5e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        rtol = 1e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-6
        self.assert_close(out, expected, rtol=rtol, atol=atol)
        self.assert_close(f.transform_matrix, expected_transform, rtol=rtol, atol=atol)

    def test_batch_random_rotation(self, device, dtype):
        # Verifies per-element random rotation invariants on a batch input:
        # p=1.0 forces every element to be transformed so the assertions don't depend on
        # how the underlying RNG happens to roll the per-element apply mask.
        torch.manual_seed(24)
        f = RandomRotation3D(degrees=45.0, p=1.0, same_on_batch=False)

        input_tensor = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                    [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                ]
            ],
            device=device,
            dtype=dtype,
        ).repeat(2, 1, 1, 1, 1)  # (2, 1, 3, 4, 4)

        atol = 5e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        rtol = 1e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-6

        torch.manual_seed(24)
        out = f(input_tensor)
        transform = f.transform_matrix

        # 1. Shape preservation
        assert out.shape == input_tensor.shape
        assert transform.shape == (2, 4, 4)

        # 2. Per-element variation: the two batch elements were rotated by *different* angles
        assert not torch.allclose(transform[0], transform[1], atol=1e-3)

        # 3. Valid rigid 3D transforms: rotation block is orthogonal, bottom row is [0,0,0,1]
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(2, 4)
        self.assert_close(transform[:, 3, :], bottom_row, rtol=rtol, atol=atol)
        rot = transform[:, :3, :3]
        rot_rt = rot @ rot.transpose(-1, -2)
        eye = torch.eye(3, device=device, dtype=dtype).expand(2, 3, 3)
        self.assert_close(rot_rt, eye, rtol=rtol, atol=atol)

        # 4. Reproducibility under fixed seed
        torch.manual_seed(24)
        out2 = f(input_tensor)
        self.assert_close(out, out2, rtol=rtol, atol=atol)
        self.assert_close(transform, f.transform_matrix, rtol=rtol, atol=atol)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation3D(degrees=40, same_on_batch=True)
        input_tensor = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 6, 1, 1)
        res = f(input_tensor)
        self.assert_close(res[0], res[1])

    def test_sequential(self, device, dtype):
        # Verifies AugmentationSequential composition for 3D rotations:
        #   Sequential(aug_a, aug_b)(x) == aug_b(aug_a(x))
        # and the composed transform matrix equals aug_b.transform_matrix @ aug_a.transform_matrix.
        # p=1.0 forces both rotations to fire so the assertion is independent of the per-element
        # apply mask. Two distinct instances are required because nn.Module dedupes children
        # passed by the same instance.
        atol = 5e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-4
        rtol = 1e-3 if (device.type == "cuda" and dtype == torch.float32) else 1e-6

        aug_a = RandomRotation3D(degrees=torch.tensor([-45.0, 90.0]), p=1.0)
        aug_b = RandomRotation3D(degrees=10.4, p=1.0)
        f = AugmentationSequential(aug_a, aug_b)

        input_tensor = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
            ],
            device=device,
            dtype=dtype,
        )  # 3 x 4 x 4

        torch.manual_seed(24)
        out_seq = f(input_tensor)
        transform_seq = f.transform_matrix

        torch.manual_seed(24)
        out_a = aug_a(input_tensor)
        transform_a = aug_a.transform_matrix
        out_manual = aug_b(out_a)
        transform_manual = aug_b.transform_matrix @ transform_a

        self.assert_close(out_seq, out_manual, rtol=rtol, atol=atol)
        self.assert_close(transform_seq, transform_manual, rtol=rtol, atol=atol)

    def test_gradcheck(self, device):
        torch.manual_seed(0)  # for random reproductibility

        input_tensor = torch.rand((3, 3, 3), device=device, dtype=torch.float64)  # 3 x 3 x 3
        self.gradcheck(RandomRotation3D(degrees=(15.0, 15.0), p=1.0), (input_tensor,))
