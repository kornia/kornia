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
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


class TestQwen3VLConfig:
    def test_defaults_populate_subconfigs(self):
        cfg = Qwen3VLConfig()
        assert isinstance(cfg.vision_config, Qwen3VLVisionConfig)
        assert isinstance(cfg.text_config, Qwen3VLTextConfig)

    def test_2b_vision_matches_official(self):
        cfg = Qwen3VLConfig.from_size("2b")
        v = cfg.vision_config
        assert v.patch_size == 16
        assert v.temporal_patch_size == 2
        assert v.spatial_merge_size == 2
        assert v.hidden_size == 1024
        assert v.depth == 24
        assert v.num_heads == 16
        assert v.intermediate_size == 4096
        assert v.hidden_act == "gelu_pytorch_tanh"
        assert v.num_position_embeddings == 2304
        assert v.out_hidden_size == 2048
        assert v.deepstack_visual_indexes == (5, 11, 17)

    @pytest.mark.parametrize("size", ["2b", "4B", "8b"])
    def test_from_size_text_hidden_matches_out_hidden(self, size):
        cfg = Qwen3VLConfig.from_size(size)
        assert cfg.text_config.hidden_size == cfg.vision_config.out_hidden_size

    def test_from_size_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown Qwen3-VL size"):
            Qwen3VLConfig.from_size("13b")
