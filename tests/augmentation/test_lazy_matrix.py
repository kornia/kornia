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
import gc
import io
import pickle
import weakref
from functools import wraps

import pytest
import torch

import kornia.augmentation as K

from testing.base import BaseTester


class ContentAwareFlip(K.RandomHorizontalFlip):
    """Flip bright images only; both pixels and coordinates depend on the input."""

    def compute_transformation(self, input, params, flags):
        matrix = super().compute_transformation(input, params, flags)
        selected = input.mean(dim=(1, 2, 3)) > 0.5
        return torch.where(selected[:, None, None], matrix, self.identity_matrix(input))

    def apply_transform(self, input, params, flags, transform=None):
        selected = input.mean(dim=(1, 2, 3)) > 0.5
        return torch.where(selected[:, None, None, None], input.flip(-1), input)


class WrappedContentAwareFlip(ContentAwareFlip):
    # functools.wraps copies function attributes; it must not make a new implementation
    # inherit a promise about whether the original implementation reads pixels.
    @wraps(K.RandomHorizontalFlip.compute_transformation)
    def compute_transformation(self, input, params, flags):
        return super().compute_transformation(input, params, flags)


class InheritedFlip(K.RandomHorizontalFlip):
    pass


def make_augmentation(name):
    factories = {
        "hflip": lambda: K.RandomHorizontalFlip(p=1.0),
        "vflip": lambda: K.RandomVerticalFlip(p=1.0),
        "invert": lambda: K.RandomInvert(p=1.0),
        "brightness": lambda: K.RandomBrightness((1.1, 1.1), p=1.0),
        "resize": lambda: K.Resize((4, 6)),
        "longest": lambda: K.LongestMaxSize(6),
        "smallest": lambda: K.SmallestMaxSize(4),
        "resized_crop": lambda: K.RandomResizedCrop((4, 6), scale=(0.8, 0.8), cropping_mode="slice"),
        "inherited_flip": lambda: InheritedFlip(p=1.0),
    }
    return factories[name]()


LAZY_OPS = ["hflip", "vflip", "invert", "brightness", "resize", "longest", "smallest", "resized_crop", "inherited_flip"]


class TestLazyMatrixState(BaseTester):
    @pytest.mark.parametrize("name", LAZY_OPS)
    def test_forward_releases_input_4482(self, name, device, dtype):
        augmentation = make_augmentation(name)
        input = torch.linspace(0, 1, 2 * 3 * 8 * 10, device=device, dtype=dtype).reshape(2, 3, 8, 10)
        reference = weakref.ref(input)
        augmentation(input)
        assert augmentation._transform_matrix is None  # matrix evaluation stays lazy
        del input
        gc.collect()
        assert reference() is None

    @pytest.mark.parametrize("serializer", ["pickle", "torch"])
    def test_serialized_size_does_not_scale_with_pixels_4482(self, serializer, device, dtype):
        sizes = []
        for side in (16, 128):
            augmentation = K.RandomHorizontalFlip(p=1.0)
            augmentation(torch.zeros(2, 3, side, side, device=device, dtype=dtype))
            if serializer == "pickle":
                payload = pickle.dumps(augmentation)
            else:
                stream = io.BytesIO()
                torch.save(augmentation, stream)
                payload = stream.getvalue()
            sizes.append(len(payload))
        # Leave room for protocol/metadata differences, but not another image's storage.
        assert sizes[1] < sizes[0] + 4096

    @pytest.mark.parametrize("augmentation_cls", [ContentAwareFlip, WrappedContentAwareFlip])
    def test_custom_pixel_dependent_matrix(self, augmentation_cls, device, dtype):
        augmentation = augmentation_cls(p=1.0)
        dark = torch.linspace(0, 0.4, 24, device=device, dtype=dtype).reshape(1, 1, 4, 6)
        input = torch.cat([dark, dark + 0.6])
        output = augmentation(input)
        assert augmentation._transform_matrix is None
        self.assert_close(output, torch.cat([dark, (dark + 0.6).flip(-1)]))
        expected = torch.tensor(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 5], [0, 1, 0], [0, 0, 1]]],
            device=device,
            dtype=dtype,
        )
        self.assert_close(augmentation.transform_matrix, expected)

    @pytest.mark.parametrize(
        "method_name",
        ["transform_tensor", "generate_transformation_matrix", "compute_transformation", "identity_matrix"],
    )
    def test_custom_matrix_path_receives_real_pixels(self, method_name, device, dtype):
        input = torch.linspace(0, 1, 24, device=device, dtype=dtype).reshape(1, 1, 4, 6)
        original = getattr(K.RandomHorizontalFlip, method_name)
        calls = []

        def implementation(self, actual, *args, **kwargs):
            calls.append(True)
            torch.testing.assert_close(actual, input)
            return original(self, actual, *args, **kwargs)

        custom_cls = type("CustomMatrixPath", (K.RandomHorizontalFlip,), {method_name: implementation})
        augmentation = custom_cls(p=0.5)
        params = augmentation.forward_parameters(input.shape)
        params["batch_prob"] = torch.ones(1, device=device)
        augmentation(input, params=params)
        matrix = augmentation.transform_matrix
        assert calls
        expected = torch.tensor([[[-1, 0, 5], [0, 1, 0], [0, 0, 1]]], device=device, dtype=dtype)
        self.assert_close(matrix, expected)

    @pytest.mark.parametrize("name", LAZY_OPS)
    def test_replay_matches_eager_matrix_path(self, name, device, dtype):
        augmentation = make_augmentation(name)
        eager = copy.deepcopy(augmentation)
        eager._compute_matrix_lazily = False
        input = torch.linspace(0, 1, 2 * 3 * 8 * 10, device=device, dtype=dtype).reshape(2, 3, 10, 8).transpose(-1, -2)
        params = augmentation.forward_parameters(input.shape)
        rng = torch.get_rng_state()
        output = augmentation(input, params=params)
        assert torch.equal(torch.get_rng_state(), rng)
        assert augmentation._transform_matrix is None
        expected_output = eager(input, params=params)
        matrix = augmentation.transform_matrix
        assert matrix is not None
        assert matrix.dtype == dtype
        assert matrix.device == input.device
        self.assert_close(output, expected_output)
        self.assert_close(matrix, eager.transform_matrix)
        assert augmentation.transform_matrix is matrix
        self.assert_close(augmentation(input, params=params), output)

    @pytest.mark.parametrize("gate", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    def test_partial_batch_matrix_and_image(self, gate, device, dtype):
        augmentation = K.RandomHorizontalFlip(p=0.5)
        input = torch.arange(48, device=device, dtype=dtype).reshape(2, 1, 4, 6)
        params = augmentation.forward_parameters(input.shape)
        params["batch_prob"] = torch.tensor(gate)  # CPU params also exercise accelerator placement.
        output = augmentation(input, params=params)
        expected_output = torch.stack([row.flip(-1) if selected else row for row, selected in zip(input, gate)])
        flip = torch.tensor([[-1, 0, 5], [0, 1, 0], [0, 0, 1]], device=device, dtype=dtype)
        identity = torch.eye(3, device=device, dtype=dtype)
        expected_matrix = torch.stack([flip if selected else identity for selected in gate])
        self.assert_close(output, expected_output)
        self.assert_close(augmentation.transform_matrix, expected_matrix)

    def test_next_call_replaces_cached_shape(self, device, dtype):
        augmentation = K.RandomHorizontalFlip(p=1.0)
        for shape in [(2, 1, 4, 6), (3, 1, 7, 9)]:
            augmentation(torch.zeros(shape, device=device, dtype=dtype))
            assert augmentation._transform_matrix is None
            expected = torch.tensor([[-1, 0, shape[-1] - 1], [0, 1, 0], [0, 0, 1]], device=device, dtype=dtype)
            self.assert_close(augmentation.transform_matrix, expected.expand(shape[0], 3, 3))

    @pytest.mark.parametrize("shape", [(4, 6), (1, 4, 6), (0, 1, 4, 6)])
    def test_unbatched_and_empty_input(self, shape, device, dtype):
        augmentation = K.RandomHorizontalFlip(p=1.0, keepdim=True)
        input = torch.zeros(shape, device=device, dtype=dtype)
        self.assert_close(augmentation(input), input)
        batch = 0 if len(shape) == 4 else 1
        expected = torch.tensor([[-1, 0, 5], [0, 1, 0], [0, 0, 1]], device=device, dtype=dtype).expand(batch, 3, 3)
        self.assert_close(augmentation.transform_matrix, expected)

    def test_input_gradient_and_graph_release(self, device, dtype):
        augmentation = K.RandomHorizontalFlip(p=1.0)
        leaf = torch.linspace(0, 1, 24, device=device, dtype=dtype).reshape(1, 1, 4, 6).requires_grad_()
        leaf_reference = weakref.ref(leaf)
        input = leaf.square()
        reference = weakref.ref(input)
        output = augmentation(input)
        output.sum().backward()
        self.assert_close(leaf.grad, 2 * leaf)
        del leaf, input, output
        gc.collect()
        assert reference() is None
        assert leaf_reference() is None
        assert augmentation.transform_matrix is not None

    def test_matrix_parameter_gradcheck(self, device):
        augmentation = K.Resize((3, 4))
        input = torch.zeros(1, 1, 5, 6, device=device, dtype=torch.float64)
        params = augmentation.forward_parameters(input.shape)
        params["src"] = params["src"].to(input)
        destination = params["dst"].to(input).requires_grad_()

        def matrix_from_destination(dst):
            augmentation(input, params={**params, "dst": dst})
            return augmentation.transform_matrix

        self.gradcheck(matrix_from_destination, (destination,))

    @pytest.mark.parametrize("mode", ["image", "silent", "skip"])
    def test_container_releases_input(self, mode, device, dtype):
        child = K.RandomHorizontalFlip(p=1.0)
        augmentation = (
            K.ImageSequential(child)
            if mode == "image"
            else K.AugmentationSequential(child, transformation_matrix_mode=mode)
        )
        input = torch.linspace(0, 1, 24, device=device, dtype=dtype).reshape(1, 1, 4, 6)
        reference = weakref.ref(input)
        output = augmentation(input)
        self.assert_close(output, input.flip(-1))
        del input, output
        gc.collect()
        assert reference() is None
        assert child.transform_matrix is not None

    def test_dynamo(self, device, dtype, torch_optimizer):
        augmentation = K.RandomHorizontalFlip(p=0.5)
        compiled = torch_optimizer(copy.deepcopy(augmentation))
        input = torch.arange(48, device=device, dtype=dtype).reshape(2, 1, 4, 6)
        params = augmentation.forward_parameters(input.shape)
        params["batch_prob"] = torch.tensor([1.0, 0.0])
        self.assert_close(compiled(input, params=params), augmentation(input, params=params))
        self.assert_close(compiled.transform_matrix, augmentation.transform_matrix)

    @pytest.mark.parametrize("serializer", ["pickle", "deepcopy", "torch"])
    def test_serialization_preserves_unread_matrix(self, serializer, device, dtype):
        augmentation = K.RandomHorizontalFlip(p=1.0)
        input = torch.arange(48, device=device, dtype=dtype).reshape(2, 1, 4, 6)
        output = augmentation(input)
        if serializer == "pickle":
            restored = pickle.loads(pickle.dumps(augmentation))  # noqa: S301
        elif serializer == "deepcopy":
            restored = copy.deepcopy(augmentation)
        else:
            stream = io.BytesIO()
            torch.save(augmentation, stream)
            stream.seek(0)
            restored = torch.load(stream, weights_only=False)
        assert restored._transform_matrix is None
        expected = torch.tensor([[-1, 0, 5], [0, 1, 0], [0, 0, 1]], device=device, dtype=dtype).expand(2, 3, 3)
        self.assert_close(restored.transform_matrix, expected)
        self.assert_close(restored(input, params=restored._params), output)
