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

from kornia.config import InstallationMode, kornia_config
from kornia.core.external import LazyLoader


class TestLazyLoaderExtra:
    """Check that a missing optional dependency points at the extra that installs it."""

    def test_missing_dependency_names_the_extra(self):
        previous_mode = kornia_config.lazyloader.installation_mode
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        try:
            loader = LazyLoader("definitely_not_a_module_xyz", extra="tracking")
            with pytest.raises(ImportError) as excinfo:
                loader.__getattr__("x")
        finally:
            kornia_config.lazyloader.installation_mode = previous_mode

        message = str(excinfo.value)
        assert "Optional dependency 'definitely_not_a_module_xyz' is not installed" in message
        assert 'pip install "kornia[tracking]"' in message

    def test_missing_dependency_without_extra_keeps_generic_message(self):
        previous_mode = kornia_config.lazyloader.installation_mode
        kornia_config.lazyloader.installation_mode = InstallationMode.RAISE
        try:
            loader = LazyLoader("definitely_not_a_module_xyz")
            with pytest.raises(ImportError) as excinfo:
                loader.__getattr__("x")
        finally:
            kornia_config.lazyloader.installation_mode = previous_mode

        message = str(excinfo.value)
        assert "Please install it to use this functionality." in message
        assert "kornia[" not in message

    @pytest.mark.parametrize(
        ("name", "extra"),
        [
            ("onnx", "onnx"),
            ("onnxruntime", "onnx"),
            ("diffusers", "sd"),
            ("boxmot", "tracking"),
            ("segmentation_models_pytorch", "segmentation"),
        ],
    )
    def test_declared_loaders_carry_their_extra(self, name, extra):
        from kornia.core import external

        assert getattr(external, name).extra == extra
