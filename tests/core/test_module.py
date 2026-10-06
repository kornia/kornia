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

import io
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip
from kornia.augmentation import ImageSequential as AugmentationImageSequential
from kornia.core.module import ImageModule, ImageModuleMixIn, ImageSequential
from kornia.io import write_image

from testing.base import DYNAMO_UNAVAILABLE_REASON, BaseTester, dynamo_is_available


class TestImageModuleMixIn:
    @pytest.fixture
    def img_module(self):
        class DummyModule(ImageModuleMixIn):
            pass

        return DummyModule()

    @pytest.fixture
    def sample_image(self):
        # Create a sample PIL image for testing
        return PILImage.fromarray(torch.randint(0, 255, (100, 100, 3)).numpy().astype(np.uint8))

    @pytest.fixture
    def sample_tensor(self):
        # Create a sample tensor for testing
        return torch.rand((3, 100, 100))

    @pytest.fixture
    def sample_numpy(self):
        # Create a sample numpy array for testing
        return torch.rand(100, 100, 3).numpy()

    def test_to_tensor_pil(self, img_module, sample_image):
        tensor = img_module.to_tensor(sample_image)
        assert isinstance(tensor, (torch.Tensor,))
        assert tensor.shape == (3, 100, 100)

    def test_to_tensor_numpy(self, img_module, sample_numpy):
        tensor = img_module.to_tensor(sample_numpy)
        assert isinstance(tensor, (torch.Tensor,))
        assert tensor.shape == (3, 100, 100)
        # A floating array is already in [0, 1]: it is transposed, not rescaled (#5207).
        torch.testing.assert_close(tensor, torch.from_numpy(sample_numpy).permute(2, 0, 1), rtol=0.0, atol=0.0)

    def test_to_tensor_tensor(self, img_module, sample_tensor):
        tensor = img_module.to_tensor(sample_tensor)
        assert tensor is sample_tensor

    def test_to_numpy_tensor(self, img_module, sample_tensor):
        array = img_module.to_numpy(sample_tensor)
        assert isinstance(array, (np.ndarray,))
        # Channels-last, the layout `to_tensor` accepts (#5207).
        assert array.shape == (100, 100, 3)

    def test_to_numpy_numpy(self, img_module, sample_numpy):
        array = img_module.to_numpy(sample_numpy)
        assert array is sample_numpy

    def test_to_pil_tensor(self, img_module, sample_tensor):
        pil_image = img_module.to_pil(sample_tensor)
        assert isinstance(pil_image, (PILImage.Image,))

    def test_to_pil_pil(self, img_module, sample_image):
        pil_image = img_module.to_pil(sample_image)
        assert pil_image is sample_image

    def test_convert_input_output(self, img_module, sample_image, sample_numpy, sample_tensor):
        @img_module.convert_input_output(output_type="numpy")
        def dummy_func(tensor):
            return tensor

        output = dummy_func(sample_image)
        assert isinstance(output, (np.ndarray,))

    def test_convert_input_output_preserves_function_metadata(self, img_module):
        def dummy_func(tensor):
            """Test function docstring."""
            return tensor

        decorated = img_module.convert_input_output()(dummy_func)

        assert decorated.__name__ == dummy_func.__name__
        assert decorated.__doc__ == dummy_func.__doc__
        assert decorated.__wrapped__ is dummy_func

    def test_convert_input_output_caches_single_output_tuple_as_tensor(self, img_module, sample_tensor):
        # A one-element tuple is returned as its element; the cache must hold the same tensor for ``show()``.
        decorated = img_module.convert_input_output(cache_output=True)(lambda tensor: (tensor,))

        output = decorated(sample_tensor)

        assert isinstance(output, torch.Tensor)
        assert isinstance(img_module._output_image, torch.Tensor)
        assert torch.equal(img_module._output_image, sample_tensor)

    def test_show(self, img_module, sample_tensor):
        img_module._output_image = sample_tensor
        pil_image = img_module.show(display=False)
        assert isinstance(pil_image, (PILImage.Image,))

    def test_save(self, img_module, sample_tensor, tmpdir):
        img_module._output_image = sample_tensor
        save_path = tmpdir.join("test_image.jpg")
        img_module.save(name=save_path)
        assert os.path.exists(save_path)

    def test_to_pil_4d_tensor_returns_list(self, img_module):
        # 4D (B, C, H, W) tensor -> list of PIL Images
        t = torch.rand(3, 3, 16, 16)
        result = img_module.to_pil(t)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(im, PILImage.Image) for im in result)

    def test_to_pil_numpy_raises(self, img_module, sample_numpy):
        with pytest.raises(NotImplementedError):
            img_module.to_pil(sample_numpy)

    def test_to_pil_1d_tensor_raises(self, img_module):
        with pytest.raises(NotImplementedError):
            img_module.to_pil(torch.rand(8))

    def test_to_numpy_pil(self, img_module, sample_image):
        arr = img_module.to_numpy(sample_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)

    def test_convert_input_output_invalid_type_raises(self, img_module, sample_tensor):
        with pytest.raises(ValueError, match="Invalid output_type"):

            @img_module.convert_input_output(output_type="invalid")
            def dummy_func(tensor):
                return tensor

    def test_convert_input_output_pil_output(self, img_module, sample_tensor):
        @img_module.convert_input_output(output_type="pil")
        def dummy_func(tensor):
            return tensor

        result = dummy_func(sample_tensor)
        assert isinstance(result, PILImage.Image)

    def test_convert_input_output_default_passes_later_strings_through(self, img_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bilinear").touch()

        @img_module.convert_input_output(output_type="pt")
        def dummy_func(image, other, mode="nearest"):
            return image, other, mode

        image = torch.zeros(3, 4, 6)
        result, other, mode = dummy_func(image, "bilinear", mode="bilinear")

        assert result is image
        assert other == "bilinear"
        assert mode == "bilinear"

    def test_convert_input_output_default_converts_arrays_and_pil_images_anywhere(self, img_module):
        @img_module.convert_input_output(output_type="pt")
        def dummy_func(a, b, c=None, d=None):
            return a, b, c, d

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        pil = PILImage.new("RGB", (6, 4))

        assert all(isinstance(t, torch.Tensor) for t in dummy_func(array, pil, c=array, d=pil))

    def test_convert_input_output_selective_input_names(self, img_module, sample_image):
        # Only convert arguments named "image", leave others unchanged
        @img_module.convert_input_output(input_names_to_handle=["image"], output_type="pt")
        def dummy_func(image, other):
            return image

        result = dummy_func(sample_image, "not_an_image")
        assert isinstance(result, torch.Tensor)

    def test_convert_input_output_default_loads_first_positional_path(self, img_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        image_path = tmp_path / "mode.png"
        PILImage.new("RGB", (6, 4)).save(image_path)

        @img_module.convert_input_output(output_type="pt")
        def dummy_func(image, mode="nearest"):
            return image, mode

        image, mode = dummy_func(str(image_path), mode="mode.png")

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 4, 6)
        assert mode == "mode.png"

    def test_convert_input_output_default_does_not_convert_keyword_file(self, img_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bilinear").touch()

        @img_module.convert_input_output(output_type="pt")
        def dummy_func(image, mode="nearest"):
            return image, mode

        image = torch.zeros(3, 4, 6)
        result, mode = dummy_func(image, mode="bilinear")

        assert result is image
        assert mode == "bilinear"

    def test_show_4d_tensor(self, img_module):
        img_module._output_image = torch.rand(4, 3, 16, 16)
        result = img_module.show(display=False)
        assert isinstance(result, PILImage.Image)

    def test_show_unsupported_backend_raises(self, img_module, sample_tensor):
        img_module._output_image = sample_tensor
        with pytest.raises(ValueError, match="Unsupported backend"):
            img_module.show(backend="matplotlib", display=False)

    def test_save_without_output_image_raises(self, img_module, tmpdir):
        img_module._output_image = None
        with pytest.raises(ValueError, match="No pre-computed images found"):
            img_module.save(name=tmpdir.join("test_image.jpg"))

    @pytest.mark.parametrize("container", [list, tuple])
    def test_store_output_image_sequence_4957(self, img_module, container, device, dtype):
        tensors = container(torch.rand(3, 4, 4, device=device, dtype=dtype, requires_grad=True) for _ in range(2))
        img_module._store_output_image(tensors, "pt")
        result = img_module._output_image
        assert isinstance(result, container)
        assert len(result) == len(tensors)
        for cached, output in zip(result, tensors):
            assert cached.device == output.device
            assert not cached.requires_grad
            assert cached.grad_fn is None
            torch.testing.assert_close(cached, output.detach())


class TestPILLookupWithoutImport(BaseTester):
    """A call that needs no PIL conversion never consults the PIL loader, whatever its other arguments are."""

    @pytest.fixture
    def pil_not_imported(self, monkeypatch):
        from kornia.core.mixin import image_module

        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"the PIL loader was consulted for {name!r}")

        # As on an install without the "image" extra: PIL.Image is not imported, and the loader must not import it.
        monkeypatch.setattr(image_module, "Image", Untouchable())
        monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    def test_container_with_data_keys(self, pil_not_imported, device, dtype):
        from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip

        image = torch.rand(1, 3, 4, 6, device=device, dtype=dtype)
        mask = torch.rand(1, 1, 4, 6, device=device, dtype=dtype)
        out_image, out_mask = AugmentationSequential(RandomHorizontalFlip(p=1.0))(
            image, mask, data_keys=["input", "mask"]
        )
        self.assert_close(out_image, image.flip(-1))
        self.assert_close(out_mask, mask.flip(-1))

    def test_image_module_with_keyword_argument(self, pil_not_imported, device, dtype):
        class Scale(ImageModule):
            def forward(self, x, scale=1.0):
                return x * scale

        x = torch.rand(3, 4, 6, device=device, dtype=dtype)
        self.assert_close(Scale()(x, scale=0.5), x * 0.5)

    def test_pil_image_is_still_converted(self):
        image = PILImage.fromarray(np.full((4, 6, 3), 255, dtype=np.uint8))
        module = ImageModuleMixIn()
        assert module._is_valid_arg(image)
        assert module._is_valid_arg([image]) is False
        assert module._is_valid_arg("not an existing path") is False


class TestImageModule:
    @pytest.fixture
    def image_module(self):
        return ImageModule()

    @pytest.fixture
    def sample_tensor(self):
        return torch.rand((3, 100, 100))

    def test_call_with_features_disabled(self, image_module, sample_tensor):
        image_module.disable_features = True
        mock_forward = MagicMock(return_value=sample_tensor)
        image_module.forward = mock_forward
        output = image_module(sample_tensor)
        assert output is sample_tensor
        mock_forward.assert_called_once()

    def test_call_with_features_enabled(self, image_module, sample_tensor):
        image_module.disable_features = False
        mock_forward = MagicMock(return_value=sample_tensor)
        image_module.forward = mock_forward
        output = image_module(sample_tensor)
        assert output is sample_tensor
        mock_forward.assert_called_once()


class TestLazyOutputCache(BaseTester):
    @pytest.fixture(params=["module", "core_sequential", "augmentation_sequential"])
    def module(self, request):
        class Sigmoid(ImageModule):
            def forward(self, x):
                return x.sigmoid()

        if request.param == "module":
            return Sigmoid()
        if request.param == "core_sequential":
            return ImageSequential(torch.nn.Sigmoid())
        return AugmentationImageSequential(torch.nn.Sigmoid())

    def test_forward_cache_stays_on_device_4957(self, module, device, dtype):
        image = torch.rand(2, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image)
        cached = module._output_image
        # Restoring an eager .cpu() must fail this assertion on CUDA/MPS.
        assert cached.device == output.device == image.device
        assert not cached.requires_grad
        assert cached.grad_fn is None
        self.assert_close(cached, output.detach())
        output.sum().backward()
        self.assert_close(image.grad, output.detach() * (1 - output.detach()))

    def test_show_save_lazy_cache_4957(self, module, device, dtype, tmp_path):
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image)
        cached = module._output_image
        working = output[0].detach().to(torch.promote_types(dtype, torch.float32))
        expected = (working.clamp(0.0, 1.0) * 255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
        rendered = module.show(display=False)
        assert isinstance(rendered, PILImage.Image)
        np.testing.assert_array_equal(np.asarray(rendered), expected)
        path = tmp_path / "cached.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected)
        assert module._output_image is cached
        assert cached.device == image.device

    def test_fresh_module_show_and_save_raise(self, module, tmp_path):
        assert module._output_image is None
        with pytest.raises(ValueError, match="No pre-computed images found"):
            module.show(display=False)
        with pytest.raises(ValueError, match="No pre-computed images found"):
            module.save(name=str(tmp_path / "empty.png"))

    def test_disabling_features_clears_output_cache(self, module, device, dtype, tmp_path):
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
        module(image)
        assert module._output_image is not None

        module.disable_features = True
        assert module._output_image is None
        module(image)
        assert module._output_image is None
        with pytest.raises(ValueError, match="No pre-computed images found"):
            module.show(display=False)
        with pytest.raises(ValueError, match="No pre-computed images found"):
            module.save(name=str(tmp_path / "disabled.png"))

    @pytest.mark.parametrize("container", ["module", "core_sequential", "augmentation_sequential"])
    def test_output_cache_is_not_serialized(self, container):
        if container == "module":
            module = _Identity()
        elif container == "core_sequential":
            module = ImageSequential(torch.nn.Identity())
        else:
            module = AugmentationImageSequential(torch.nn.Identity())

        def serialized_size():
            buffer = io.BytesIO()
            torch.save(module, buffer)
            return buffer.tell()

        before = serialized_size()
        output = module(torch.rand(4, 3, 64, 64))
        assert "_output_image" not in module.__getstate__()
        assert serialized_size() - before < output.numel() * output.element_size()
        buffer = io.BytesIO()
        torch.save(module, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        assert not hasattr(restored, "_output_image")

    @pytest.mark.parametrize("shape,expected_size", [((3, 1, 5), (5, 1)), ((3, 5, 1), (1, 5))])
    def test_show_and_save_preserve_unit_spatial_dimensions(self, shape, expected_size, tmp_path):
        module = _Identity()
        module(torch.rand(shape))
        image = module.show(display=False)
        assert image.mode == "RGB"
        assert image.size == expected_size
        path = tmp_path / "single-row-or-column.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            assert saved.mode == "RGB"
            assert saved.size == expected_size

    def test_two_dimensional_output_has_clear_show_and_save_error(self, tmp_path):
        module = ImageModuleMixIn()
        module._output_image = torch.rand(4, 5)
        with pytest.raises(ValueError, match="Expected a 3D or 4D image tensor"):
            module.show(display=False)
        with pytest.raises(ValueError, match="Expected a 3D or 4D image tensor"):
            module.save(name=str(tmp_path / "2d.png"))

    def test_bfloat16_output_can_be_shown_and_converted_to_numpy(self):
        module = _Identity()
        image = torch.rand(3, 4, 6, dtype=torch.bfloat16)
        output = module(image, output_type="numpy")
        assert output.dtype == np.float32
        np.testing.assert_array_equal(output, image.float().permute(1, 2, 0).numpy())
        rendered = module.show(display=False)
        assert rendered.mode == "RGB"
        assert rendered.size == (6, 4)

    @pytest.mark.parametrize("output_type", ["numpy", "pil"])
    def test_requested_output_conversion_4957(self, module, output_type, device, dtype):
        if dtype == torch.bfloat16 and output_type == "numpy":
            pytest.skip("NumPy does not support bfloat16")
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        output = module(image, output_type=output_type)
        expected = image.sigmoid().detach().cpu()
        if output_type == "numpy":
            assert isinstance(output, np.ndarray)
            np.testing.assert_array_equal(output, expected.permute(0, 2, 3, 1).numpy())
        else:
            assert isinstance(output, list)
            assert len(output) == 1
            assert isinstance(output[0], PILImage.Image)
            working = expected[0].to(torch.promote_types(dtype, torch.float32))
            rendered = (working.clamp(0.0, 1.0) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(np.asarray(output[0]), rendered)

    @pytest.mark.parametrize("output_type", ["numpy", "pil"])
    def test_converted_output_can_be_shown_and_saved_4964(self, module, output_type, device, dtype, tmp_path):
        if dtype == torch.bfloat16 and output_type == "numpy":
            pytest.skip("NumPy does not support bfloat16")
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        module(image, output_type=output_type)
        cached = module._output_image
        assert isinstance(cached, torch.Tensor)
        assert cached.device == image.device
        assert cached.dtype == image.dtype
        assert cached.grad_fn is None
        assert not cached.requires_grad
        self.assert_close(cached, image.sigmoid().detach())
        working = cached[0].to(torch.promote_types(dtype, torch.float32))
        expected = (working.clamp(0.0, 1.0) * 255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(np.asarray(module.show(display=False)), expected)
        path = tmp_path / "converted.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected)
        assert module._output_image is cached

    @pytest.mark.skipif(not dynamo_is_available(), reason=DYNAMO_UNAVAILABLE_REASON)
    def test_export_preserves_cache_4957(self, device, dtype):
        class CacheModule(torch.nn.Module, ImageModuleMixIn):
            def forward(self, x):
                output = x.sigmoid()
                self._store_output_image(output, "pt")
                return output

        module = CacheModule()
        image = torch.rand(1, 3, 6, 8, device=device, dtype=dtype)
        expected = module(image)
        cached = module._output_image
        exported = torch.export.export(module, (image,), strict=True)
        self.assert_close(exported.module()(image), expected)
        assert module._output_image is cached


class _Identity(ImageModule):
    def forward(self, x):
        return x


# The #5209 repro row: truncated (0.999), a half step (0.5), above the range (1.1), below it (-0.1), and the ends.
_ROW = [0.999, 0.5, 1.1, -0.1, 0.0, 1.0]
_ROW_UINT8 = [255, 128, 255, 0, 0, 255]


class TestImageModuleConversions(BaseTester):
    """Value and layout contract of the ``ImageModule`` input and output conversions."""

    @pytest.mark.parametrize("half", [torch.float16, torch.bfloat16])
    def test_half_images_round_like_their_exact_value_5209(self, half):
        # Every half value in [0, 1]: the 8-bit pixel is round(v * 255) of the exact value, so the product must not be
        # rounded to the half dtype first (float16 0.0058823 * 255 is 1.49998, which float16 rounds up to 1.5).
        from kornia.core.mixin.image_module import _to_uint8_image

        bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
        values = bits.view(half)
        values = values[values.isfinite() & (values >= 0) & (values <= 1)]
        expected = (values.double() * 255).round().to(torch.uint8)
        assert torch.equal(_to_uint8_image(values), expected)

    @pytest.mark.parametrize("np_dtype", [np.uint8, np.uint16, np.int8, np.int16, np.int32])
    def test_to_tensor_scales_integer_numpy_by_dtype_max_5207(self, np_dtype):
        info = np.iinfo(np_dtype)
        values = [0, info.max // 2, info.max] if info.min == 0 else [info.min, -1, 0, info.max // 2, info.max]
        pixel = np.array(values, dtype=np_dtype)
        image = np.broadcast_to(pixel, (2, 4, len(values))).copy()  # (H, W, C), every pixel is `pixel`
        out = _Identity().to_tensor(image)
        expected = torch.tensor(pixel.astype(np.float64) / info.max, dtype=torch.float32).view(-1, 1, 1)
        self.assert_close(out, expected.expand(len(values), 2, 4))
        if info.min < 0:  # a signed minimum lands just below -1: int8 -128 -> -128 / 127
            assert out[0, 0, 0].item() == pytest.approx(info.min / info.max)

    @pytest.mark.parametrize("default_dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("np_dtype", [np.uint16, np.int32])
    def test_to_tensor_scales_in_float32_under_a_half_default_dtype_5207(self, np_dtype, default_dtype):
        # 65535 and 2**31 - 1 overflow float16 (largest finite 65504): converting before dividing gave inf.
        info = np.iinfo(np_dtype)
        pixel = np.array([0, 1, 65504, 65535, info.max], dtype=np_dtype)
        image = np.broadcast_to(pixel, (2, 4, len(pixel))).copy()  # (H, W, C), every pixel is `pixel`
        previous = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            out = _Identity().to_tensor(image)
        finally:
            torch.set_default_dtype(previous)
        assert out.dtype == default_dtype
        assert torch.isfinite(out).all()
        expected = torch.tensor(pixel.astype(np.float64) / info.max).to(default_dtype).view(-1, 1, 1)
        self.assert_close(out, expected.expand(len(pixel), 2, 4), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("np_dtype", [np.float16, np.float32, np.float64])
    def test_to_tensor_passes_floating_numpy_through_5207(self, np_dtype):
        image = np.random.default_rng(0).random((2, 4, 3)).astype(np_dtype)
        out = _Identity().to_tensor(image)
        self.assert_close(out, torch.from_numpy(image).permute(2, 0, 1), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("container", [ImageModule, ImageSequential, AugmentationImageSequential])
    @pytest.mark.parametrize("state", ["parameter", "buffer"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
    def test_to_tensor_uses_module_device_and_dtype_5207(self, container, state, dtype):
        module = container()
        reference = torch.empty((), device="meta", dtype=dtype)
        if state == "parameter":
            module.register_parameter("reference", torch.nn.Parameter(reference))
        else:
            module.register_buffer("reference", reference)

        image = np.full((2, 4, 3), 0.5, dtype=np.float32)
        out = module.to_tensor(image)
        assert out.device == torch.device("meta")
        assert out.dtype == dtype

    def test_to_tensor_maps_bool_numpy_to_zero_one_5207(self):
        mask = np.array([[True, False, True], [False, True, False]])
        out = _Identity().to_tensor(mask)
        self.assert_close(out, torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "shape, expected_shape",
        [((4, 6), (1, 4, 6)), ((4, 6, 1), (1, 4, 6)), ((4, 6, 3), (3, 4, 6)), ((2, 4, 6, 3), (2, 3, 4, 6))],
    )
    def test_to_tensor_numpy_layouts_5207(self, shape, expected_shape):
        image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        channels_first = image[None] if image.ndim == 2 else np.moveaxis(image, -1, -3)
        out = _Identity().to_tensor(image)
        assert out.shape == expected_shape
        self.assert_close(out, torch.from_numpy(channels_first.astype(np.float32) / 255))

    def test_to_tensor_path_5207(self, tmp_path):
        path = tmp_path / "rgb.png"
        PILImage.new("RGB", (6, 4), (255, 51, 0)).save(path)
        out = _Identity().to_tensor(str(path))
        self.assert_close(out, torch.tensor([1.0, 0.2, 0.0]).view(3, 1, 1).expand(3, 4, 6))

    def test_to_tensor_16_bit_path_5207(self, tmp_path):
        path = tmp_path / "rgb16.png"
        write_image(str(path), torch.tensor([65535, 13107, 0], dtype=torch.uint16).view(3, 1, 1).expand(3, 4, 6))
        out = _Identity().to_tensor(str(path))
        self.assert_close(out, torch.tensor([1.0, 0.2, 0.0]).view(3, 1, 1).expand(3, 4, 6))

    def test_numpy_output_round_trips_through_numpy_input_5207(self):
        module = _Identity()
        image = np.random.default_rng(0).integers(0, 256, (4, 6, 3), dtype=np.uint8)
        first = module(image, output_type="numpy")
        # The numpy output is channels-last like the numpy input, in the [0, 1] range of the converted tensor.
        assert first.shape == image.shape
        np.testing.assert_array_equal(first, (image / 255).astype(np.float32))
        second = module(first, output_type="numpy")
        np.testing.assert_array_equal(second, first)

    @pytest.mark.parametrize("shape", [(3, 4, 6), (1, 4, 6), (2, 3, 4, 6), (2, 1, 4, 6)])
    def test_to_numpy_is_the_inverse_of_to_tensor_5207(self, shape, device, dtype):
        if dtype == torch.bfloat16:
            pytest.skip("NumPy has no bfloat16")
        module = _Identity()
        image = torch.rand(shape, device=device, dtype=dtype)
        array = module.to_numpy(image)
        assert array.shape == (*shape[:-3], *shape[-2:], shape[-3])
        self.assert_close(module.to_tensor(array), image.cpu(), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "mode, fill, expected",
        [
            ("L", 51, [0.2]),
            ("I;16", 65535, [1.0]),
            ("I;16B", 65535, [1.0]),
            ("I", 2**31 - 1, [1.0]),
            ("F", 0.25, [0.25]),
            ("1", 1, [1.0]),
            ("LA", (51, 255), [0.2, 1.0]),
            ("RGB", (255, 51, 0), [1.0, 0.2, 0.0]),
            ("RGBA", (255, 51, 0, 255), [1.0, 0.2, 0.0, 1.0]),
        ],
    )
    def test_to_tensor_pil_modes_5208(self, mode, fill, expected):
        out = _Identity().to_tensor(PILImage.new(mode, (6, 4), fill))
        expected = torch.tensor(expected).view(-1, 1, 1).expand(len(expected), 4, 6)
        self.assert_close(out, expected)

    @pytest.mark.parametrize(
        "mode, fill, transparency, expected",
        [
            ("P", 1, None, [1.0, 0.0, 0.0]),
            ("P", 1, 0, [1.0, 0.0, 0.0, 1.0]),
            ("PA", (1, 51), None, [1.0, 0.0, 0.0, 0.2]),
        ],
    )
    def test_to_tensor_pil_palette_converts_to_colors_5208(self, mode, fill, transparency, expected):
        image = PILImage.new(mode, (6, 4), fill)
        image.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))  # index 1 is red
        if transparency is not None:
            image.info["transparency"] = transparency
        out = _Identity().to_tensor(image)
        self.assert_close(out, torch.tensor(expected).view(-1, 1, 1).expand(len(expected), 4, 6))

    def test_to_pil_one_channel_is_mode_l_5208(self, device, dtype):
        module = _Identity()
        image = torch.tensor([0.0, 0.2, 1.0], device=device, dtype=dtype).view(1, 1, 3).expand(1, 2, 3)
        pil = module.to_pil(image)
        assert pil.mode == "L"
        np.testing.assert_array_equal(np.asarray(pil), [[0, 51, 255], [0, 51, 255]])
        batch = module.to_pil(image[None].expand(2, 1, 2, 3))
        assert [im.mode for im in batch] == ["L", "L"]
        (converted,) = module(image[None], output_type="pil")
        assert converted.mode == "L"
        np.testing.assert_array_equal(np.asarray(converted), [[0, 51, 255], [0, 51, 255]])

    def test_to_pil_rejects_other_inputs_with_a_message_5208(self):
        module = _Identity()
        with pytest.raises(NotImplementedError, match=r"\(C, H, W\)"):
            module.to_pil(torch.rand(4, 6))
        with pytest.raises(NotImplementedError, match="NumPy"):
            module.to_pil(np.zeros((4, 6, 3), np.uint8))

    def _row_image(self, device, dtype):
        return torch.tensor(_ROW, device=device, dtype=dtype).view(1, 1, 6).expand(3, 2, 6).contiguous()

    def test_to_pil_clamps_and_rounds_5209(self, device, dtype):
        pil = _Identity().to_pil(self._row_image(device, dtype))
        assert pil.mode == "RGB"
        pixels = np.asarray(pil)
        for channel in range(3):
            np.testing.assert_array_equal(pixels[:, :, channel], [_ROW_UINT8, _ROW_UINT8])

    def test_show_clamps_and_rounds_5209(self, device, dtype):
        module = _Identity()
        module(self._row_image(device, dtype))
        shown = module.show(display=False)
        np.testing.assert_array_equal(np.asarray(shown)[:, :, 0], [_ROW_UINT8, _ROW_UINT8])

    def test_save_clamps_and_rounds_5209(self, device, dtype, tmp_path):
        module = _Identity()
        module(self._row_image(device, dtype))
        path = tmp_path / "row.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            np.testing.assert_array_equal(np.asarray(saved)[:, :, 0], [_ROW_UINT8, _ROW_UINT8])

    def test_integer_image_is_not_rescaled_5209(self, device, tmp_path):
        module = _Identity()
        image = torch.full((3, 2, 2), 200, dtype=torch.uint8, device=device)
        assert np.asarray(module.to_pil(image)).tolist() == [[[200] * 3] * 2] * 2
        (converted,) = module(image[None], output_type="pil")
        assert np.asarray(converted).tolist() == [[[200] * 3] * 2] * 2
        module(image)
        assert np.asarray(module.show(display=False)).tolist() == [[[200] * 3] * 2] * 2
        path = tmp_path / "u8.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            assert np.asarray(saved).tolist() == [[[200] * 3] * 2] * 2

    @pytest.mark.parametrize(
        "torch_dtype, channels", [(torch.uint16, 3), (torch.int32, 3), (torch.int64, 3), (torch.int64, 1)]
    )
    def test_to_pil_rejects_integer_images_pil_cannot_store_5209(self, torch_dtype, channels):
        image = torch.zeros(channels, 2, 2, dtype=torch_dtype)
        with pytest.raises(NotImplementedError, match=f"{channels}-channel {torch_dtype}"):
            _Identity().to_pil(image)

    def test_to_pil_show_save_agree_on_float16_ties_5209(self, device, tmp_path):
        # x * 255 lands on exact .5 ties in float16 (0.00196 -> 0.5). All three methods round on the CPU, half to even:
        # MPS on torch 2.5.1 rounds ties away from zero, so rounding on the device made `to_pil` differ from the others.
        generator = torch.Generator().manual_seed(0)
        image = torch.rand(3, 32, 32, generator=generator).to(device=device, dtype=torch.float16)
        image[:, 0, 0] = 0.00196
        module = _Identity()
        module(image)
        pil = np.asarray(module.to_pil(image))
        assert pil[0, 0].tolist() == [0, 0, 0]
        np.testing.assert_array_equal(pil, np.asarray(module.show(display=False)))
        path = tmp_path / "ties.png"
        module.save(name=str(path))
        with PILImage.open(path) as saved:
            np.testing.assert_array_equal(pil, np.asarray(saved))


class TestTupleOutputCache(BaseTester):
    @pytest.mark.parametrize("container", [ImageModule, ImageSequential])
    @pytest.mark.parametrize("count", [1, 2])
    @pytest.mark.parametrize("output_type", ["pt", "numpy", "pil"])
    def test_tuple_conversion_and_cache(self, container, count, output_type, device, dtype):
        if dtype == torch.bfloat16 and output_type == "numpy":
            pytest.skip("NumPy does not support bfloat16")

        class TupleModule(container):
            def forward(self, x):
                return tuple(x.sigmoid() for _ in range(count))

        module = TupleModule()
        image = torch.rand(3, 6, 8, device=device, dtype=dtype, requires_grad=True)
        result = module(image, output_type=output_type)
        cached = module._output_image
        if count == 1:
            result, cached = [result], [cached]
        else:
            assert isinstance(result, list)
            assert isinstance(cached, list)
        assert len(result) == len(cached) == count
        expected = image.sigmoid().detach()
        for output, tensor in zip(result, cached):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device == image.device
            assert not tensor.requires_grad
            self.assert_close(tensor, expected)
            if output_type == "pt":
                assert output.requires_grad
                self.assert_close(output, expected)
            elif output_type == "numpy":
                np.testing.assert_array_equal(output, expected.cpu().permute(1, 2, 0).numpy())
            else:
                assert isinstance(output, PILImage.Image)
                working = expected.cpu().to(torch.promote_types(dtype, torch.float32))
                rendered = (working.clamp(0.0, 1.0) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(np.asarray(output), rendered)


class TestNamedInputConversion(BaseTester):
    @pytest.mark.parametrize("keyword", [False, True])
    def test_module_forward_names_5206(self, keyword):
        class Select(ImageModule):
            def forward(self, image, other, *, scale=1.0):
                assert isinstance(image, torch.Tensor)
                assert other is untouched
                return image * scale

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        untouched = array.copy()
        module = Select()
        calls = []
        module.register_forward_hook(lambda *args: calls.append(True))
        kwargs = {"other": untouched, "scale": 0.5, "input_names_to_handle": ["image"]}
        output = module(image=array, **kwargs) if keyword else module(array, **kwargs)
        self.assert_close(output, torch.full((3, 4, 6), 0.5))
        assert calls == [True]

    @pytest.mark.parametrize("kind", ["core", "augmentation"])
    def test_sequential_forward_name_5206(self, kind):
        factory = ImageSequential if kind == "core" else AugmentationImageSequential
        module = factory(torch.nn.Identity())
        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        output = module(array, input_names_to_handle=["input"])
        assert isinstance(output, torch.Tensor)
        assert output.shape[-3:] == (3, 4, 6)
        self.assert_close(output, torch.ones_like(output))

    def test_bound_decorator_5206(self):
        class Methods:
            def select(self, image, other):
                assert other is untouched
                return image

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        untouched = array.copy()
        decorated = ImageModuleMixIn().convert_input_output(["image"])(Methods().select)
        output = decorated(array, untouched)
        assert isinstance(output, torch.Tensor)
        self.assert_close(output, torch.ones(3, 4, 6))

    def test_variadic_decorator_5206(self):
        module = ImageModuleMixIn()

        @module.convert_input_output(["images", "options"])
        def select(*images, **options):
            assert all(isinstance(x, torch.Tensor) for x in images)
            assert isinstance(options["image"], torch.Tensor)
            return images[0]

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        self.assert_close(select(array, array, image=array), torch.ones(3, 4, 6))

    def test_keyword_name_in_variadic_options_5206(self):
        @ImageModuleMixIn().convert_input_output(["image"])
        def select(**options):
            assert options["other"] is untouched
            return options["image"]

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        untouched = array.copy()
        self.assert_close(select(image=array, other=untouched), torch.ones(3, 4, 6))

    def test_positional_only_and_keyword_only_5206(self):
        @ImageModuleMixIn().convert_input_output(["image", "mask"])
        def select(image, /, *, mask):
            return image + mask

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        self.assert_close(select(array, mask=array), torch.full((3, 4, 6), 2.0))

    def test_augmentation_variadic_forward_5206(self):
        module = AugmentationSequential(RandomHorizontalFlip(p=1.0), data_keys=["input"])
        array = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        expected = torch.from_numpy(array).permute(2, 0, 1).float().div(255).flip(-1)
        output = module(array, input_names_to_handle=["args"])
        self.assert_close(output[0], expected)

    @pytest.mark.parametrize("keyword", [False, True])
    def test_hooks_see_the_callers_argument_split_5206(self, keyword):
        class Select(ImageModule):
            def forward(self, image, other=None, **options):
                return image

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        module = Select()
        seen = []
        module.register_forward_pre_hook(
            lambda _, args, kwargs: seen.append((len(args), sorted(kwargs), type(kwargs.get("image")))),
            with_kwargs=True,
        )
        if keyword:
            module(image=array, other=1, extra=array, input_names_to_handle=["image", "extra"])
            assert seen == [(0, ["extra", "image", "other"], torch.Tensor)]
        else:
            module(array, other=1, extra=array, input_names_to_handle=["image", "extra"])
            assert seen == [(1, ["extra", "other"], type(None))]

    def test_arguments_that_do_not_bind_raise_the_calls_own_error_5206(self):
        class Select(ImageModule):
            def forward(self, image):
                return image

        array = np.full((4, 6, 3), 255, dtype=np.uint8)
        with pytest.raises(TypeError, match="forward"):
            Select()(array, array, input_names_to_handle=["image"])

    def test_keyword_named_like_a_positional_only_parameter_5206(self):
        @ImageModuleMixIn().convert_input_output(["image"])
        def select(image, /, **options):
            return image, options["image"]

        image, option = select(np.full((4, 6, 3), 255, dtype=np.uint8), image=np.zeros((4, 6, 3), dtype=np.uint8))
        self.assert_close(image, torch.ones(3, 4, 6))
        self.assert_close(option, torch.zeros(3, 4, 6))
