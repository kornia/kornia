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

from __future__ import annotations

import copy
import pickle

import pytest
import torch
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend

import kornia.augmentation as K

from testing.base import BaseTester


class TestAugmentationCompile(BaseTester):
    def test_compile_distinct_classes(self, device, dtype, torch_optimizer):
        # More classes than Dynamo's default per-code-object cache limit (#4658).
        augmentations = [
            K.RandomHorizontalFlip(p=1),
            K.RandomVerticalFlip(p=1),
            K.RandomInvert(p=1),
            K.RandomGrayscale(p=1),
            K.RandomBrightness(p=1),
            K.RandomContrast(p=1),
            K.RandomGamma(p=1),
            K.RandomSharpness(p=1),
            K.RandomSolarize(p=1),
        ]
        input = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        compiled = []
        for aug in augmentations:
            # Isolate shared entry-frame caching here. Crop and CLAHE random-path
            # tests separately trace parameter sampling and check graph reuse.
            params = aug.forward_parameters(input.shape)
            counter = CompileCounter()
            fn = torch_optimizer(aug, backend=counter, fullgraph=True)
            self.assert_close(fn(input, params=params), aug(input, params=params))
            assert counter.frame_count > 0
            compiled.append((aug, fn, params, counter, counter.frame_count))
        for aug, fn, params, counter, frames in compiled:
            self.assert_close(fn(input, params=params), aug(input, params=params))
            assert counter.frame_count == frames

    @pytest.mark.parametrize(
        "resample,align_corners",
        [("bilinear", True), ("bilinear", False), ("bicubic", True), ("bicubic", False), ("nearest", None)],
    )
    @pytest.mark.parametrize("size", [(5, 7), (1, 1)])
    def test_compile_resized_crop_replay(self, device, dtype, torch_optimizer, resample, align_corners, size):
        aug = K.RandomResizedCrop(size, resample=resample, align_corners=align_corners)
        input = torch.rand(2, 3, 12, 15, device=device, dtype=dtype, requires_grad=True)
        counter = CompileCounter()
        fn = torch_optimizer(aug, backend=counter, fullgraph=True)
        # Include singleton crops, up/downsampling, borders and different boxes within a batch.
        boxes = [
            ((0, 0, 15, 12), (2, 3, 7, 6)),
            ((4, 2, 1, 1), (1, 0, 11, 10)),
            ((2, 5, 10, 4), (7, 1, 4, 9)),
            ((0, 0, 5, 7), (3, 2, 7, 5)),
        ]
        for batch_boxes in boxes:
            params = aug.forward_parameters(input.shape)
            params["src"] = torch.tensor(
                [[[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]] for x, y, w, h in batch_boxes],
                device=device,
                dtype=dtype,
            )
            expected = aug(input, params=params)
            actual = fn(input, params=params)
            self.assert_close(actual, expected)
            weights = torch.rand_like(actual)
            actual_grad = torch.autograd.grad(actual, input, weights)[0]
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
                # CUDA half interpolate backward rounds atomic updates. Compare with
                # float32 accumulation, as in the real-backend replay tests below.
                reference = input.detach().float().requires_grad_()
                expected_grad = torch.autograd.grad(aug(reference, params=params), reference, weights.float())[0].to(
                    dtype
                )
            else:
                expected_grad = torch.autograd.grad(expected, input, weights)[0]
            self.assert_close(actual_grad, expected_grad)
        assert counter.frame_count == 1

    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_compile_resized_crop_random(self, device, dtype, torch_optimizer, same_on_batch):
        aug = K.RandomResizedCrop((8, 8), same_on_batch=same_on_batch)
        input = torch.rand(2, 3, 16, 16, device=device, dtype=dtype)
        counter = CompileCounter()
        fn = torch_optimizer(aug, backend=counter, fullgraph=True)
        outputs = [fn(input) for _ in range(20)]
        assert all(output.shape == (2, 3, 8, 8) for output in outputs)
        assert any(not torch.equal(outputs[0], output) for output in outputs[1:])
        assert counter.frame_count == 1

    def test_forward_code_qualname(self):
        for cls in (K.RandomHorizontalFlip, K.RandomVerticalFlip):
            assert cls.forward.__code__.co_qualname == f"{cls.__qualname__}.forward"

    @pytest.mark.parametrize("backend", ["eager", "inductor"])
    def test_compile_nearest_dynamic_shape(self, device, dtype, torch_optimizer, backend):
        aug = K.RandomResizedCrop((46, 22), resample="nearest", align_corners=None)
        counter = CompileCounter() if backend == "eager" else CompileCounterWithBackend(backend)
        fn = torch_optimizer(aug, backend=counter, fullgraph=True, dynamic=True)
        frames = None
        for height, width in [(14, 26), (18, 30), (22, 34)]:
            input = torch.rand(2, 3, height, width, device=device, dtype=dtype)
            params = aug.forward_parameters(input.shape)
            params["src"] = (
                torch.tensor(
                    [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
                    device=device,
                    dtype=dtype,
                )
                .expand(2, -1, -1)
                .clone()
            )
            self.assert_close(fn(input, params=params), aug(input, params=params), atol=0, rtol=0)
            if frames is None:
                # Cold Inductor can invoke the backend twice during initial dynamic
                # compilation. Changing spatial sizes must reuse the resulting graph.
                frames = counter.frame_count
                assert frames > 0
            assert counter.frame_count == frames

    def test_forward_copy_and_pickle(self, device, dtype):
        aug = K.RandomHorizontalFlip(p=1)
        input = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        for clone in (copy.deepcopy(aug), pickle.loads(pickle.dumps(aug))):  # noqa: S301
            self.assert_close(clone(input), input.flip(-1))

    @pytest.mark.parametrize(
        "resample,align_corners",
        [("bilinear", True), ("bilinear", False), ("bicubic", True), ("bicubic", False), ("nearest", None)],
    )
    def test_compile_resized_crop_backend(self, device, dtype, torch_optimizer, resample, align_corners):
        aug = K.RandomResizedCrop((9, 11), resample=resample, align_corners=align_corners)
        input = torch.rand(2, 3, 16, 17, device=device, dtype=dtype, requires_grad=True)
        params = aug.forward_parameters(input.shape)
        expected = aug(input, params=params)
        actual = torch_optimizer(aug, fullgraph=True)(input, params=params)
        self.assert_close(actual, expected)
        weights = torch.rand_like(actual)
        actual_grad = torch.autograd.grad(actual, input, weights)[0]
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            # CUDA interpolate accumulates half gradients with reduced precision. Use
            # its float32 result as the oracle for our float32 tap accumulation.
            reference = input.detach().float().requires_grad_()
            expected_grad = torch.autograd.grad(aug(reference, params=params), reference, weights.float())[0].to(dtype)
        else:
            expected_grad = torch.autograd.grad(expected, input, weights)[0]
        self.assert_close(actual_grad, expected_grad)
        # Exercise sampling and resizing in one graph, not just parameter replay.
        fresh = K.RandomResizedCrop((9, 11), resample=resample, align_corners=align_corners)
        fn = torch_optimizer(fresh, fullgraph=True)
        assert fn(input).shape == fn(input).shape == (2, 3, 9, 11)

    @pytest.mark.parametrize("backend", ["eager", "inductor"])
    def test_compile_nearest_upsample_gradient(self, device, dtype, torch_optimizer, backend):
        torch.manual_seed(17)
        aug = K.RandomResizedCrop((32, 32), resample="nearest", align_corners=None)
        input = torch.rand(1, 1, 2, 2, device=device, dtype=dtype, requires_grad=True)
        params = aug.forward_parameters(input.shape)
        params["src"] = torch.zeros(1, 4, 2, device=device, dtype=dtype)
        actual = torch_optimizer(aug, backend=backend, fullgraph=True)(input, params=params)
        self.assert_close(actual, aug(input, params=params), atol=0, rtol=0)
        weights = torch.rand_like(actual)
        # All 1024 output gradients accumulate into one pixel. Half atomics can
        # saturate here; use float32 accumulation for both half dtypes as the oracle.
        reference = input.detach().to(torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype)
        reference.requires_grad_()
        expected_grad = torch.autograd.grad(aug(reference, params=params), reference, weights.to(reference))[0]
        self.assert_close(torch.autograd.grad(actual, input, weights)[0], expected_grad.to(dtype))

    def test_compile_nearest_rounding(self, device, dtype, torch_optimizer):
        aug = K.RandomResizedCrop((46, 22), resample="nearest", align_corners=None)
        input = torch.arange(14 * 26, device=device, dtype=dtype).reshape(1, 1, 14, 26)
        params = aug.forward_parameters(input.shape)
        params["src"] = torch.tensor([[[0, 0], [25, 0], [25, 13], [0, 13]]], device=device, dtype=dtype)
        self.assert_close(
            torch_optimizer(aug, fullgraph=True)(input, params=params), aug(input, params=params), atol=0, rtol=0
        )

    def test_compile_nearest_without_resize(self, device, dtype, torch_optimizer):
        aug = K.RandomResizedCrop((4, 4), scale=(1, 1), ratio=(1, 1), resample="nearest")
        input = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        self.assert_close(torch_optimizer(aug, fullgraph=True)(input), aug(input))

    def test_compile_inherited_override(self, device, dtype, torch_optimizer):
        class CustomFlip(K.RandomHorizontalFlip):
            def forward(self, input, params=None, *, offset=2):
                return super().forward(input, params=params) + offset

        class InheritedFlip(CustomFlip):
            pass

        input = torch.rand(2, 3, 8, 8, device=device, dtype=dtype)
        aug = InheritedFlip(p=1)
        self.assert_close(torch_optimizer(aug, fullgraph=True)(input, offset=3), input.flip(-1) + 3)
        self.assert_close(copy.deepcopy(aug)(input), input.flip(-1) + 2)

    @pytest.mark.parametrize("channels,channels_last", [(1, False), (4, False), (4, True)])
    def test_compile_nearest_large_output(self, device, dtype, torch_optimizer, channels, channels_last):
        aug = K.RandomResizedCrop((1, 210), resample="nearest", align_corners=None)
        input = torch.rand(2, channels, 2, 465, device=device, dtype=dtype)
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        fn = torch_optimizer(aug, fullgraph=True)
        # Large float64 CPU outputs use a different ATen nearest kernel. Full-width
        # cropped rows can change between channels-last and strided batch layouts.
        for boxes in [
            ((0, 0, 465, 2), (0, 0, 465, 2)),
            ((0, 0, 465, 1), (0, 0, 465, 1)),
            ((0, 0, 465, 1), (0, 1, 465, 1)),
            ((1, 0, 463, 2), (0, 0, 465, 2)),
        ]:
            params = aug.forward_parameters(input.shape)
            params["src"] = torch.tensor(
                [[[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]] for x, y, w, h in boxes],
                device=device,
                dtype=dtype,
            )
            self.assert_close(fn(input, params=params), aug(input, params=params), atol=0, rtol=0)

    def test_inherited_static_forward(self, device, dtype):
        class StaticFlip(K.RandomHorizontalFlip):
            @staticmethod
            def forward(input):
                return input + 1

        class InheritedFlip(StaticFlip):
            pass

        input = torch.rand(1, 1, 2, 2, device=device, dtype=dtype)
        self.assert_close(InheritedFlip()(input), input + 1)


class TestRandomCropCompile(BaseTester):
    @pytest.mark.parametrize("padding_mode", ["constant", "reflect", "replicate"])
    def test_compile_crop_replay(self, device, dtype, torch_optimizer, padding_mode):
        aug = K.RandomCrop((5, 7), padding=(2, 1, 3, 2), padding_mode=padding_mode)
        input = torch.rand(2, 3, 12, 15, device=device, dtype=dtype, requires_grad=True)
        counter = CompileCounter()
        fn = torch_optimizer(aug, backend=counter)
        for offset in range(10):
            params = aug.forward_parameters(input.shape)
            params["src"] = torch.tensor(
                [
                    [[offset, 0], [offset + 6, 0], [offset + 6, 4], [offset, 4]],
                    [[1, 2], [3, 2], [3, 5], [1, 5]],
                ],
                device=device,
                dtype=dtype,
            )
            expected = aug(input, params=params)
            actual = fn(input, params=params)
            self.assert_close(actual, expected)
            weights = torch.rand_like(actual)
            actual_grad = torch.autograd.grad(actual, input, weights)[0]
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
                # CUDA half interpolate backward rounds atomic updates. Compare with
                # float32 accumulation, as in the real-backend replay tests below.
                reference = input.detach().float().requires_grad_()
                expected_grad = torch.autograd.grad(aug(reference, params=params), reference, weights.float())[0].to(
                    dtype
                )
            else:
                expected_grad = torch.autograd.grad(expected, input, weights)[0]
            self.assert_close(actual_grad, expected_grad)
            if offset == 1:
                frames = counter.frame_count
        assert counter.frame_count == frames

    @pytest.mark.parametrize("same_on_batch", [False, True])
    def test_compile_crop_random(self, device, dtype, torch_optimizer, same_on_batch):
        aug = K.RandomCrop((32, 32), same_on_batch=same_on_batch)
        input = torch.rand(32, 3, 64, 64, device=device, dtype=dtype)
        counter = CompileCounter()
        fn = torch_optimizer(aug, backend=counter)
        outputs = []
        for i in range(40):
            outputs.append(fn(input))
            if i == 1:
                frames = counter.frame_count
        assert all(output.shape == (32, 3, 32, 32) for output in outputs)
        assert any(not torch.equal(outputs[0], output) for output in outputs[1:])
        assert counter.frame_count == frames

    @pytest.mark.parametrize("size,pad_if_needed", [((7, 9), False), ((19, 23), False), ((19, 23), True)])
    def test_compile_crop_backend(self, device, dtype, torch_optimizer, size, pad_if_needed):
        aug = K.RandomCrop(size, pad_if_needed=pad_if_needed, padding=1)
        input = torch.rand(2, 3, 12, 15, device=device, dtype=dtype, requires_grad=True)
        fn = torch_optimizer(aug)
        for _ in range(3):
            params = aug.forward_parameters(input.shape)
            expected = aug(input, params=params)
            actual = fn(input, params=params)
            self.assert_close(actual, expected)
            weights = torch.rand_like(actual)
            actual_grad = torch.autograd.grad(actual, input, weights)[0]
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
                # Compare against opmath accumulation, as for compiled resized crops above.
                reference = input.detach().float().requires_grad_()
                expected_grad = torch.autograd.grad(aug(reference, params=params), reference, weights.float())[0].to(
                    dtype
                )
            else:
                expected_grad = torch.autograd.grad(expected, input, weights)[0]
            self.assert_close(actual_grad, expected_grad)

    @pytest.mark.parametrize("augmentation", [K.RandomCrop, K.RandomResizedCrop])
    @pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
    @pytest.mark.parametrize("resample", ["bilinear", "bicubic"])
    def test_compile_crop_without_resize_nonfinite(
        self, device, dtype, torch_optimizer, augmentation, nonfinite, resample
    ):
        aug = augmentation((3, 3), resample=resample)
        input = torch.arange(16, device=device, dtype=dtype).reshape(1, 1, 4, 4)
        input[..., 1, 1] = nonfinite
        params = aug.forward_parameters(input.shape)
        params["src"] = torch.tensor([[[0, 0], [2, 0], [2, 2], [0, 2]]], device=device, dtype=dtype)
        actual = torch_optimizer(aug)(input, params=params)
        expected = aug(input, params=params)
        self.assert_close(actual.isnan(), expected.isnan())
        self.assert_close(actual.isinf(), expected.isinf())
        self.assert_close(actual.nan_to_num(), expected.nan_to_num(), atol=0, rtol=0)

    @pytest.mark.parametrize("size,align_corners", [(41, False), (42, True)])
    def test_compile_identity_resize_coordinates(self, device, dtype, torch_optimizer, size, align_corners):
        aug = K.RandomResizedCrop((size, size), align_corners=align_corners)
        input = torch.rand(1, 1, size, size, device=device, dtype=dtype)
        params = aug.forward_parameters(input.shape)
        params["src"] = torch.tensor(
            [[[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]], device=device, dtype=dtype
        )
        self.assert_close(torch_optimizer(aug, fullgraph=True)(input, params=params), input, atol=0, rtol=0)
