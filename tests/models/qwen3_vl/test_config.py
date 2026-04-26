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

from kornia.models.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLProjectorConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


class TestQwen3VLConfig:
    def test_defaults_populate_subconfigs(self):
        cfg = Qwen3VLConfig()
        assert isinstance(cfg.vision_config, Qwen3VLVisionConfig)
        assert isinstance(cfg.projector_config, Qwen3VLProjectorConfig)
        assert isinstance(cfg.text_config, Qwen3VLTextConfig)
        assert cfg.projector_config.input_dim == cfg.vision_config.hidden_size

    @pytest.mark.parametrize("size", ["2b", "4B", "8b"])
    def test_from_size_returns_full_config(self, size):
        cfg = Qwen3VLConfig.from_size(size)
        assert cfg.text_config.hidden_size == cfg.projector_config.output_dim

    def test_from_size_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown Qwen3-VL size"):
            Qwen3VLConfig.from_size("13b")
