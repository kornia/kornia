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
