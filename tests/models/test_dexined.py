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

from kornia.filters.dexined import DexiNed as FilterDexiNed
from kornia.models.dexined import DexiNed

from testing.base import BaseTester


class TestDexiNed(BaseTester):
    def test_smoke(self, device, dtype):
        img = torch.rand(2, 3, 32, 32, device=device, dtype=dtype)
        net = DexiNed(pretrained=False).to(device, dtype)
        feat = net.get_features(img)
        assert len(feat) == 6
        out = net(img)
        assert out.shape == (2, 1, 32, 32)

    @pytest.mark.slow
    @pytest.mark.parametrize("data", ["dexined"], indirect=True)
    def test_inference(self, device, dtype, data):
        model = DexiNed(pretrained=False)
        model.load_state_dict(data, strict=True)
        model = model.to(device, dtype)
        model.eval()

        img = torch.tensor([[[[0.0, 255.0, 0.0], [0.0, 255.0, 0.0], [0.0, 255.0, 0.0]]]], device=device, dtype=dtype)
        img = img.repeat(1, 3, 1, 1)

        expect = torch.tensor(
            [[[[-0.3709, 0.0519, -0.2839], [0.0627, 0.6587, -0.1276], [-0.1840, -0.3917, -0.8240]]]],
            device=device,
            dtype=dtype,
        )

        out = model(img)
        self.assert_close(out, expect, atol=1e-3, rtol=1e-2)

    @pytest.mark.skip(reason="DexiNed do not compile with dynamo.")
    def test_dynamo(self, device, dtype, torch_optimizer): ...


@pytest.mark.parametrize("cls", [FilterDexiNed, DexiNed], ids=["filters", "models"])
class TestDexiNedLoadFromFile:
    """``load_from_file`` reads a local file itself; the hub cache is keyed by base name (#5477)."""

    @staticmethod
    def _hub_with_a_cached_decoy(tmp_path, monkeypatch, name):
        hub = tmp_path / "hub"
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
        (hub / "checkpoints").mkdir(parents=True)
        torch.save({"w": torch.zeros(3)}, hub / "checkpoints" / name)

    @staticmethod
    def _record_load_state_dict(model, monkeypatch):
        loaded = []
        monkeypatch.setattr(
            model, "load_state_dict", lambda state_dict, strict=True: loaded.append((state_dict, strict))
        )
        return loaded

    @pytest.mark.parametrize("spelling", ["absolute", "home"])
    def test_a_local_file_is_loaded_even_when_the_cache_holds_its_name(self, cls, spelling, tmp_path, monkeypatch):
        self._hub_with_a_cached_decoy(tmp_path, monkeypatch, "dexined.pth")
        local = tmp_path / "weights" / "dexined.pth"
        local.parent.mkdir()
        torch.save({"w": torch.ones(3)}, local)
        if spelling == "home":
            monkeypatch.setenv("HOME", str(tmp_path))
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
            path_file = "~/weights/dexined.pth"
        else:
            path_file = str(local)
        model = cls(pretrained=False).train()
        loaded = self._record_load_state_dict(model, monkeypatch)

        model.load_from_file(path_file)

        assert len(loaded) == 1
        state_dict, strict = loaded[0]
        assert strict is True
        assert torch.equal(state_dict["w"], torch.ones(3))
        assert not model.training

    def test_a_string_that_names_no_file_is_still_treated_as_a_url(self, cls, tmp_path, monkeypatch):
        self._hub_with_a_cached_decoy(tmp_path, monkeypatch, "dexined.pth")
        model = cls(pretrained=False)
        loaded = self._record_load_state_dict(model, monkeypatch)

        with pytest.raises(ValueError, match="scheme"):
            model.load_from_file(str(tmp_path / "missing" / "dexined.pth"))

        assert loaded == []
