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

import hashlib

import pytest

import kornia.feature.lightglue_onnx.utils.download as download_utils


def test_download_routes_through_core_helper(monkeypatch, tmp_path):
    call = {}

    def fake_download(url, **kwargs):
        call["url"] = url
        call.update(kwargs)
        return str(tmp_path / kwargs["file_name"])

    monkeypatch.setattr(download_utils.core_download, "download_file_from_url", fake_download)

    path = download_utils.download_onnx_from_url(
        "https://example.org/model.onnx", model_dir=str(tmp_path), progress=False, timeout=2.5
    )

    assert path == str(tmp_path / "model.onnx")
    assert call == {
        "url": "https://example.org/model.onnx",
        "file_name": "model.onnx",
        "model_dir": str(tmp_path),
        "progress": False,
        "validate": None,
        "timeout": 2.5,
    }


@pytest.mark.parametrize("valid", [True, False])
def test_check_hash_is_preserved_as_core_validator(monkeypatch, tmp_path, valid):
    expected = b"model bytes"
    payload = expected if valid else b"wrong bytes"
    prefix = hashlib.sha256(expected).hexdigest()[:8]
    file_name = f"model-{prefix}.onnx"

    def fake_download(_url, **kwargs):
        path = tmp_path / kwargs["file_name"]
        path.write_bytes(payload)
        kwargs["validate"](str(path))
        return str(path)

    monkeypatch.setattr(download_utils.core_download, "download_file_from_url", fake_download)

    if valid:
        assert download_utils.download_onnx_from_url(
            f"https://example.org/{file_name}", model_dir=str(tmp_path), check_hash=True
        ) == str(tmp_path / file_name)
    else:
        with pytest.raises(RuntimeError, match="invalid hash value"):
            download_utils.download_onnx_from_url(
                f"https://example.org/{file_name}", model_dir=str(tmp_path), check_hash=True
            )
