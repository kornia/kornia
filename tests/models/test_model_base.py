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

import os
from unittest.mock import patch

import pytest
import torch

from kornia.models.base import ModelBaseMixin


class DummyMixin(ModelBaseMixin):
    name = "dummy"


class TestModelBaseMixinTensorToType:
    def test_torch_output_returns_tensor_unchanged(self):
        mixin = DummyMixin()
        t = torch.rand(1, 3, 8, 8)
        out = mixin._tensor_to_type(t, "torch")
        assert out is t

    def test_torch_output_returns_list_unchanged(self):
        mixin = DummyMixin()
        tensors = [torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)]
        out = mixin._tensor_to_type(tensors, "torch")
        assert out is tensors

    def test_pil_output_single_tensor(self):
        mixin = DummyMixin()
        # (C, H, W) tensor in [0, 1]
        t = torch.rand(3, 16, 16)
        out = mixin._tensor_to_type(t, "pil")
        # tensor_to_image returns a PIL Image or ndarray depending on implementation
        assert out is not None

    def test_pil_output_list_of_tensors(self):
        mixin = DummyMixin()
        tensors = [torch.rand(3, 16, 16), torch.rand(3, 16, 16)]
        out = mixin._tensor_to_type(tensors, "pil")
        assert isinstance(out, list)
        assert len(out) == 2

    def test_unsupported_output_type_raises(self):
        mixin = DummyMixin()
        t = torch.rand(1, 3, 8, 8)
        with pytest.raises(RuntimeError, match=r"Output type.*is not supported"):
            mixin._tensor_to_type(t, "numpy")


class TestModelBaseMixinSave:
    def test_save_single_tensor_calls_write_image_once(self, tmp_path):
        mixin = DummyMixin()
        t = torch.rand(3, 8, 8)
        with patch("kornia.models.base.write_image") as mock_write:
            mixin.save(t, str(tmp_path))
            assert mock_write.call_count == 1
            # path is the first positional arg (matches write_image(path_file, image, ...))
            saved_path = mock_write.call_args[0][0]
            assert os.path.normcase(str(tmp_path)) in os.path.normcase(saved_path)

    def test_save_list_of_tensors_calls_write_image_per_item(self, tmp_path):
        mixin = DummyMixin()
        tensors = [torch.rand(3, 8, 8), torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
        with patch("kornia.models.base.write_image") as mock_write:
            mixin.save(tensors, str(tmp_path))
            assert mock_write.call_count == 3

    def test_save_creates_directory(self, tmp_path):
        mixin = DummyMixin()
        t = torch.rand(3, 8, 8)
        new_dir = str(tmp_path / "new_subdir")
        assert not os.path.exists(new_dir)
        with patch("kornia.models.base.write_image"):
            mixin.save(t, new_dir)
        assert os.path.exists(new_dir)


class TestModelBaseMixinSaveOutputs:
    def test_save_outputs_with_explicit_dir_single_tensor(self, tmp_path):
        mixin = DummyMixin()
        t = torch.rand(3, 8, 8)
        with patch("kornia.models.base.write_image") as mock_write:
            mixin._save_outputs(t, directory=str(tmp_path), suffix="_test")
            assert mock_write.call_count == 1
            # path is the first positional arg; verify suffix appears in filename
            saved_path = mock_write.call_args[0][0]
            assert "_test_" in saved_path

    def test_save_outputs_with_explicit_dir_list(self, tmp_path):
        mixin = DummyMixin()
        tensors = [torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
        with patch("kornia.models.base.write_image") as mock_write:
            mixin._save_outputs(tensors, directory=str(tmp_path))
            assert mock_write.call_count == 2

    def test_save_outputs_none_dir_creates_default(self, tmp_path, monkeypatch):
        # Run from tmp_path so we don't pollute the repo
        monkeypatch.chdir(tmp_path)
        mixin = DummyMixin()
        t = torch.rand(3, 8, 8)
        with patch("kornia.models.base.write_image"):
            mixin._save_outputs(t, directory=None)
        kornia_outputs = tmp_path / "kornia_outputs"
        assert kornia_outputs.exists()
        subdirs = list(kornia_outputs.iterdir())
        assert len(subdirs) == 1
        assert subdirs[0].name.startswith("dummy")


class TestModelBaseMixinSaveWritesRealFiles:
    """`save` against the real `write_image`, not a mock.

    Every other test in this file patches `write_image`, which is why #4322
    shipped: the mock accepts any dtype and any rank, so nothing checked that
    what the containers hand it is something it can actually write. `visualize`
    returns float images and the containers document batched input, and
    `write_image` accepts neither -- PNG is uint8/uint16 only, and the rank has
    to be (3, H, W), (1, H, W) or (H, W).
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
    def test_a_float_visualization_is_written(self, tmp_path, dtype):
        mixin = DummyMixin()
        mixin.save(torch.rand(3, 8, 8, dtype=dtype), str(tmp_path))
        written = list(tmp_path.iterdir())
        assert len(written) == 1, f"expected one file, got {written}"
        assert written[0].stat().st_size > 0

    def test_a_batch_is_written_as_one_file_per_item(self, tmp_path):
        """The containers document (B, 3, H, W); write_image takes (3, H, W)."""
        mixin = DummyMixin()
        mixin.save(torch.rand(4, 3, 8, 8), str(tmp_path))
        assert len(list(tmp_path.iterdir())) == 4

    def test_a_batch_of_one_is_still_indexed(self, tmp_path):
        """So a caller does not have to guess between name.png and name_0.png."""
        mixin = DummyMixin()
        mixin.save(torch.rand(1, 3, 8, 8), str(tmp_path))
        written = list(tmp_path.iterdir())
        assert len(written) == 1
        assert written[0].name.endswith("_0.png")

    def test_the_written_file_is_a_readable_image(self, tmp_path):
        """Writing something unreadable would satisfy a file-count check."""
        from kornia.io import ImageLoadType, load_image

        mixin = DummyMixin()
        mixin.save(torch.rand(3, 8, 8), str(tmp_path))
        written = next(iter(tmp_path.iterdir()))
        image = load_image(str(written), ImageLoadType.UNCHANGED)
        assert image.shape == (3, 8, 8)
        assert image.dtype == torch.uint8

    def test_values_survive_the_conversion(self, tmp_path):
        """A float in [0, 1] must land on the matching 0-255 level."""
        from kornia.io import ImageLoadType, load_image

        mixin = DummyMixin()
        source = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]]).repeat(3, 1, 1)
        mixin.save(source, str(tmp_path))
        written = next(iter(tmp_path.iterdir()))
        image = load_image(str(written), ImageLoadType.UNCHANGED)
        assert image[0].tolist() == [[0, 255], [128, 64]]

    def test_out_of_range_values_are_clamped_not_wrapped(self, tmp_path):
        """Without the clamp, 1.5 * 255 overflows uint8 and comes back dark."""
        from kornia.io import ImageLoadType, load_image

        mixin = DummyMixin()
        source = torch.tensor([[[1.5, -0.5]]]).repeat(3, 1, 1)
        mixin.save(source, str(tmp_path))
        written = next(iter(tmp_path.iterdir()))
        image = load_image(str(written), ImageLoadType.UNCHANGED)
        assert image[0].tolist() == [[255, 0]]

    def test_an_integer_image_is_passed_through(self, tmp_path):
        from kornia.io import ImageLoadType, load_image

        mixin = DummyMixin()
        source = torch.tensor([[[0, 255], [7, 64]]], dtype=torch.uint8).repeat(3, 1, 1)
        mixin.save(source, str(tmp_path))
        written = next(iter(tmp_path.iterdir()))
        image = load_image(str(written), ImageLoadType.UNCHANGED)
        assert image[0].tolist() == [[0, 255], [7, 64]]

    def test_save_outputs_writes_real_files_too(self, tmp_path):
        mixin = DummyMixin()
        mixin._save_outputs(torch.rand(2, 3, 8, 8), directory=str(tmp_path), suffix="_mask")
        written = sorted(p.name for p in tmp_path.iterdir())
        assert len(written) == 2
        assert all("_mask_" in name for name in written)
