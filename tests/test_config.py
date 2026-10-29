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

import os

import torch

from kornia.config import KorniaConfig, kornia_config


class TestHubDirectoryDefaults:
    def test_defaults_are_under_the_torch_hub_dir(self):
        root = os.path.abspath(os.path.join(torch.hub.get_dir(), "kornia"))
        config = KorniaConfig()
        assert config.hub_cache_dir == root
        assert config.hub_models_dir == os.path.join(root, "models")
        assert config.hub_onnx_dir == os.path.join(root, "onnx_models")

    def test_defaults_are_absolute(self):
        for path in (kornia_config.hub_cache_dir, kornia_config.hub_models_dir, kornia_config.hub_onnx_dir):
            assert os.path.isabs(path)

    def test_defaults_do_not_depend_on_the_working_directory(self, tmp_path, monkeypatch):
        # the test session pins torch.hub to a relative "weights/" (conftest.py); use torch's own default instead
        monkeypatch.setattr(torch.hub, "_hub_dir", None)
        monkeypatch.delenv("TORCH_HOME", raising=False)
        before = KorniaConfig()
        monkeypatch.chdir(tmp_path)
        after = KorniaConfig()
        assert (after.hub_cache_dir, after.hub_models_dir, after.hub_onnx_dir) == (
            before.hub_cache_dir,
            before.hub_models_dir,
            before.hub_onnx_dir,
        )

    def test_follows_torch_home(self, tmp_path, monkeypatch):
        # torch.hub.get_dir() reads TORCH_HOME on every call unless torch.hub.set_dir was used
        monkeypatch.setattr(torch.hub, "_hub_dir", None)
        monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_home"))
        config = KorniaConfig()
        assert config.hub_cache_dir == os.path.join(str(tmp_path / "torch_home"), "hub", "kornia")

    def test_relative_torch_home_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.hub, "_hub_dir", None)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TORCH_HOME", "relative_home")
        config = KorniaConfig()
        assert config.hub_cache_dir == os.path.join(str(tmp_path), "relative_home", "hub", "kornia")

    def test_fields_can_still_be_set(self, tmp_path):
        config = KorniaConfig(hub_models_dir=str(tmp_path / "m"), hub_onnx_dir=str(tmp_path / "o"))
        assert config.hub_models_dir == str(tmp_path / "m")
        assert config.hub_onnx_dir == str(tmp_path / "o")
        config.hub_cache_dir = str(tmp_path / "c")
        assert config.hub_cache_dir == str(tmp_path / "c")
