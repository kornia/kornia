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

import sys
import warnings
from unittest.mock import call, patch

import pytest

from kornia.core.download import hf_url, load_state_dict_from_url


class TestHfUrl:
    def test_format(self) -> None:
        assert hf_url("hardnet", "HardNetPP.pth") == (
            "https://huggingface.co/kornia/hardnet/resolve/main/HardNetPP.pth"
        )

    def test_subdirectory(self) -> None:
        url = hf_url("loftr", "loftr_outdoor.ckpt")
        assert url.startswith("https://huggingface.co/kornia/loftr/resolve/main/")


class TestLoadStateDictFromUrl:
    _SD = {"weight": 1}
    _MOCK_TARGET = "kornia.core.download.torch.hub.load_state_dict_from_url"

    def test_single_url_success(self) -> None:
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url("http://example.com/model.pth")
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth")

    def test_list_single_url_success(self) -> None:
        # A single-element list behaves like a plain str — no file_name injection
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            result = load_state_dict_from_url(["http://example.com/model.pth"])
        assert result == self._SD
        mock.assert_called_once_with("http://example.com/model.pth")

    def test_fallback_on_failure(self) -> None:
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = load_state_dict_from_url([primary, fallback])

        assert result == self._SD
        assert mock.call_count == 2
        assert any("primary down" in str(warning.message) for warning in w)

    def test_all_fail_raises_runtime_error(self) -> None:
        with patch(self._MOCK_TARGET, side_effect=OSError("down")):
            with pytest.raises(RuntimeError, match="Failed to load weights from all 2 source"):
                load_state_dict_from_url(["http://a.com/m.pth", "http://b.com/m.pth"])

    def test_file_name_pinned_to_primary(self) -> None:
        primary = "http://primary.example.com/weights-abc123.pth"
        fallback = "http://fallback.example.com/weights.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_state_dict_from_url([primary, fallback])

        # fallback call must carry the primary's filename, not the fallback's
        fallback_call = mock.call_args_list[1]
        assert fallback_call == call(fallback, file_name="weights-abc123.pth")

    def test_explicit_file_name_not_overridden(self) -> None:
        primary = "http://primary.example.com/model.pth"
        fallback = "http://fallback.example.com/model.pth"

        def side_effect(url: str, **kwargs: object) -> dict:
            if url == primary:
                raise OSError("primary down")
            return self._SD

        with patch(self._MOCK_TARGET, side_effect=side_effect) as mock:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                load_state_dict_from_url([primary, fallback], file_name="custom.pth")

        for c in mock.call_args_list:
            assert c.kwargs.get("file_name") == "custom.pth"

    def test_kwargs_forwarded(self) -> None:
        with patch(self._MOCK_TARGET, return_value=self._SD) as mock:
            load_state_dict_from_url("http://example.com/model.pth", map_location="cpu")
        mock.assert_called_once_with("http://example.com/model.pth", map_location="cpu")


class TestProgressGoesToStderr:
    """torch.hub writes its 'Downloading: ...' line to stdout (torch 2.x).

    Status output on stdout corrupts callers that treat stdout as data. The most
    visible victim is ``pytest --doctest-modules kornia/``: the line is captured as
    unexpected example output and fails any example that downloads on a cold cache
    (#4005). The wrapper keeps that chatter on stderr instead.
    """

    _MOCK_TARGET = "kornia.core.download.torch.hub.load_state_dict_from_url"
    _CHATTER = 'Downloading: "http://example.com/model.pth" to /cache/model.pth'

    def _noisy(self, url: str, **kwargs: object) -> dict:
        print(self._CHATTER)
        return {"weight": 1}

    def test_stdout_stays_clean(self, capsys) -> None:
        with patch(self._MOCK_TARGET, side_effect=self._noisy):
            result = load_state_dict_from_url("http://example.com/model.pth")

        captured = capsys.readouterr()
        assert result == {"weight": 1}
        assert captured.out == ""
        assert self._CHATTER in captured.err

    def test_stdout_restored_after_call(self, capsys) -> None:
        original = sys.stdout
        with patch(self._MOCK_TARGET, side_effect=self._noisy):
            load_state_dict_from_url("http://example.com/model.pth")
        assert sys.stdout is original

    def test_stdout_restored_after_failure(self, capsys) -> None:
        original = sys.stdout
        with patch(self._MOCK_TARGET, side_effect=OSError("down")):
            with pytest.raises(RuntimeError):
                load_state_dict_from_url("http://example.com/model.pth")
        assert sys.stdout is original
