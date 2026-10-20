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
import os
import urllib
from pathlib import Path

import pytest

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")

from kornia.onnx.utils import ONNXLoader  # noqa: E402

# the directory that a URL whose query is "download=1" is cached in
_Q_DOWNLOAD_1 = "%3F" + hashlib.sha256(b"download=1").hexdigest()[:16]


class TestONNXLoader:
    def test_get_file_path(self):
        # Test getting local file path for caching
        model_name = "some_model"
        expected_path = os.path.join(".kornia_hub", "some_model.onnx")
        assert ONNXLoader._get_file_path(model_name, None, suffix=".onnx") == expected_path

    @pytest.mark.parametrize("name", ["a/b", "a\\b"])
    def test_get_file_path_on_windows_splits_on_both_separators(self, name, monkeypatch):
        # hf:// names and URL paths use "/", and a name built with os.path.join uses "\\" on Windows: both split
        import ntpath
        from types import SimpleNamespace

        import kornia.onnx.download

        monkeypatch.setattr(kornia.onnx.download, "os", SimpleNamespace(sep="\\", path=ntpath))
        assert ONNXLoader._get_file_path(name, "cache", suffix=".onnx") == ntpath.join("cache", "a", "b.onnx")

    @pytest.mark.parametrize("name", ["C:x", "C:\\abs\\x", "d:/x", "a/C:", "\\\\server\\share\\x"])
    def test_get_file_path_on_windows_rejects_a_drive(self, name, monkeypatch):
        # os.path.join("cache", "C:x") is "C:x", a path on drive C:, not a name inside cache
        import ntpath
        from types import SimpleNamespace

        import kornia.onnx.download

        monkeypatch.setattr(kornia.onnx.download, "os", SimpleNamespace(sep="\\", path=ntpath))
        with pytest.raises(ValueError, match="relative path inside the cache directory, without a drive"):
            ONNXLoader._get_file_path(name, "cache", suffix=".onnx")

    def test_load_model_local(self):
        from unittest import mock

        from onnx import ModelProto

        with mock.patch("onnx.load") as mock_onnx_load, mock.patch("os.path.exists") as mock_exists:
            model_name = "local_model.onnx"
            mock_exists.return_value = True

            # Simulate onnx.load returning a dummy ModelProto
            mock_model = mock.Mock(spec=ModelProto)
            mock_onnx_load.return_value = mock_model

            model = ONNXLoader.load_model(model_name)
            assert model == mock_model
            mock_onnx_load.assert_called_once_with(model_name)

    def test_load_model_download(self, tmp_path):
        from unittest import mock

        from onnx import ModelProto

        with (
            mock.patch.object(ONNXLoader, "download") as mock_download,
            mock.patch("onnx.load") as mock_onnx_load,
        ):
            model_name = "hf://operators/some_model"
            mock_model = mock.Mock(spec=ModelProto)
            mock_onnx_load.return_value = mock_model

            model = ONNXLoader.load_model(model_name, cache_dir=str(tmp_path))
            assert model == mock_model
            mock_download.assert_called_once_with(
                "https://huggingface.co/kornia/ONNX_models/resolve/main/operators/some_model.onnx",
                str(tmp_path / "some_model.onnx"),
                download_if_not_exists=True,
            )
            mock_onnx_load.assert_called_once_with(str(tmp_path / "some_model.onnx"))

    @pytest.mark.parametrize("absolute", [False, True])
    def test_load_model_hf_default_cache_dir(self, absolute, monkeypatch, tmp_path):
        # without cache_dir, an hf:// model is cached under <hub_onnx_dir>/<folder>/, also for an absolute hub_onnx_dir
        from unittest import mock

        from kornia.config import kornia_config

        hub_dir = str(tmp_path / "onnx_models") if absolute else os.path.join("rel", "onnx_models")
        monkeypatch.setattr(kornia_config, "hub_onnx_dir", hub_dir)

        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
            ONNXLoader.load_model("hf://operators/some_model")

        mock_download.assert_called_once_with(
            "https://huggingface.co/kornia/ONNX_models/resolve/main/operators/some_model.onnx",
            os.path.join(hub_dir, "operators", "some_model.onnx"),
            download_if_not_exists=True,
        )

    @pytest.mark.parametrize("explicit_cache_dir", [False, True])
    def test_load_model_hf_name_keeps_every_path_component(self, explicit_cache_dir, monkeypatch, tmp_path):
        # #5902: hf://models/a/b and hf://models/a/c used to share one cache file, <dir>/a.onnx, and so did their
        # .onnx_data weights; each must keep its own path below the folder, with the weights beside the model
        from unittest import mock

        from kornia.config import kornia_config

        hub_dir = str(tmp_path / "onnx_models")
        monkeypatch.setattr(kornia_config, "hub_onnx_dir", hub_dir)
        cache_dir = str(tmp_path / "mine") if explicit_cache_dir else os.path.join(hub_dir, "models")
        kwargs = {"cache_dir": cache_dir} if explicit_cache_dir else {}

        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load") as mock_onnx_load:
            ONNXLoader.load_model("hf://models/a/b", with_data=True, **kwargs)
            ONNXLoader.load_model("hf://models/a/c", **kwargs)

        base = "https://huggingface.co/kornia/ONNX_models/resolve/main/models/a"
        assert mock_download.call_args_list == [
            mock.call(f"{base}/b.onnx", os.path.join(cache_dir, "a", "b.onnx"), download_if_not_exists=True),
            mock.call(f"{base}/b.onnx_data", os.path.join(cache_dir, "a", "b.onnx_data"), download_if_not_exists=True),
            mock.call(f"{base}/c.onnx", os.path.join(cache_dir, "a", "c.onnx"), download_if_not_exists=True),
        ]
        assert [c.args[0] for c in mock_onnx_load.call_args_list] == [
            os.path.join(cache_dir, "a", "b.onnx"),
            os.path.join(cache_dir, "a", "c.onnx"),
        ]

    def test_load_model_url_cache_path_follows_the_url(self, monkeypatch, tmp_path):
        # #5902: a URL was cached under its last component only, query included (m.onnx?download=1.onnx), so URLs
        # that end in the same file name shared one entry. The entry follows the scheme, the host and the path; a query
        # gets a directory named after its hash, shared by the model and its weights.
        from unittest import mock

        from kornia.config import kornia_config

        hub_dir = str(tmp_path / "onnx_models")
        monkeypatch.setattr(kornia_config, "hub_onnx_dir", hub_dir)
        urls = [
            "https://example.invalid/x/m.onnx?download=1",
            "https://example.invalid/y/m.onnx",
            "https://other.invalid/x/m.onnx",
        ]

        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
            ONNXLoader.load_model(urls[0], with_data=True)
            for url in urls[1:]:
                ONNXLoader.load_model(url)

        q = _Q_DOWNLOAD_1
        assert mock_download.call_args_list == [
            mock.call(
                urls[0],
                os.path.join(hub_dir, "https", "example.invalid", "x", q, "m.onnx"),
                download_if_not_exists=True,
            ),
            mock.call(
                "https://example.invalid/x/m.onnx_data?download=1",
                os.path.join(hub_dir, "https", "example.invalid", "x", q, "m.onnx_data"),
                download_if_not_exists=True,
            ),
            mock.call(
                urls[1], os.path.join(hub_dir, "https", "example.invalid", "y", "m.onnx"), download_if_not_exists=True
            ),
            mock.call(
                urls[2], os.path.join(hub_dir, "https", "other.invalid", "x", "m.onnx"), download_if_not_exists=True
            ),
        ]

    def test_load_model_url_without_onnx_extension_fetches_its_weights_beside_it(self, monkeypatch, tmp_path):
        from unittest import mock

        from kornia.config import kornia_config

        hub_dir = str(tmp_path / "onnx_models")
        monkeypatch.setattr(kornia_config, "hub_onnx_dir", hub_dir)
        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
            ONNXLoader.load_model("https://example.invalid/x/model", with_data=True)
        directory = os.path.join(hub_dir, "https", "example.invalid", "x")
        assert [c.args[:2] for c in mock_download.call_args_list] == [
            ("https://example.invalid/x/model", os.path.join(directory, "model.onnx")),
            ("https://example.invalid/x/model.onnx_data", os.path.join(directory, "model.onnx_data")),
        ]

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.invalid/x/m.onnx#part", "https/example.invalid/x/m.onnx"),
            ("https://Example.INVALID/X/m.onnx", "https/example.invalid/X/m.onnx"),
            ("https://user:secret@example.invalid/m.onnx", "https/example.invalid/m.onnx"),
            # the ":" before a port, and those of an IPv6 address, are encoded as "%" is, so no host name can be
            # mistaken for a host with a port; the brackets keep "[::1]:8000" apart from "[::1:8000]"
            ("http://127.0.0.1:8000/m.onnx", "http/127.0.0.1%3A8000/m.onnx"),
            ("http://a_8000/m.onnx", "http/a_8000/m.onnx"),
            ("http://[::1]/m.onnx", "http/[%3A%3A1]/m.onnx"),
            ("http://[::1]:8000/m.onnx", "http/[%3A%3A1]%3A8000/m.onnx"),
            ("http://[::1:8000]/m.onnx", "http/[%3A%3A1%3A8000]/m.onnx"),
            ("http://__1_8000/m.onnx", "http/__1_8000/m.onnx"),
            ("https://example.invalid/a/../b/./m.onnx", "https/example.invalid/b/m.onnx"),
            ("https://example.invalid/../../m.onnx", "https/example.invalid/m.onnx"),
            ("https://example.invalid//x//m.onnx", "https/example.invalid/x/m.onnx"),
            ("https://example.invalid/x/model", "https/example.invalid/x/model.onnx"),
            ("https://example.invalid/x/m.onnx?download=1", f"https/example.invalid/x/{_Q_DOWNLOAD_1}/m.onnx"),
            # a directory of the path never becomes the query's: its "%" is encoded
            (
                f"https://example.invalid/x/{_Q_DOWNLOAD_1}/m.onnx",
                f"https/example.invalid/x/%25{_Q_DOWNLOAD_1[1:]}/m.onnx",
            ),
            # a directory keeps no character that Windows refuses (":" would also make "C:" a drive), and "%" is
            # encoded too, so that "x:y" and "x%3Ay" stay two directories
            ("https://example.invalid/x:y/a b/m.onnx", "https/example.invalid/x%3Ay/a%20b/m.onnx"),
            ("https://example.invalid/x%3Ay/m.onnx", "https/example.invalid/x%253Ay/m.onnx"),
            ("https://example.invalid/C:/m.onnx", "https/example.invalid/C%3A/m.onnx"),
            ('https://example.invalid/<a>|"*\\/m.onnx', "https/example.invalid/%3Ca%3E%7C%22%2A%5C/m.onnx"),
        ],
    )
    @pytest.mark.parametrize("cache_dir", ["absent", None, ""])
    def test_load_model_url_default_cache_path(self, url, expected, cache_dir, monkeypatch, tmp_path):
        # the entry stays inside the hub directory: no credentials, ":" or ".." reach the path; an empty cache_dir
        # means the default directory, as it does for hf:// names
        from unittest import mock

        from kornia.config import kornia_config

        hub_dir = str(tmp_path / "onnx_models")
        monkeypatch.setattr(kornia_config, "hub_onnx_dir", hub_dir)
        kwargs = {} if cache_dir == "absent" else {"cache_dir": cache_dir}
        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
            ONNXLoader.load_model(url, **kwargs)

        mock_download.assert_called_once_with(
            url, os.path.join(hub_dir, *expected.split("/")), download_if_not_exists=True
        )

    def test_load_model_url_with_cache_dir_follows_the_url_below_it(self, tmp_path):
        # an explicit cache_dir is the root the URL's entry is stored under, as the default directory is: every
        # onnx-community repository names its model onnx/model.onnx, and two of them must not share one file
        from unittest import mock

        cache_dir = str(tmp_path / "mine")
        urls = [f"https://huggingface.co/onnx-community/{m}/resolve/main/onnx/model.onnx" for m in ("small", "large")]
        with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
            for url in urls:
                ONNXLoader.load_model(url, with_data=True, cache_dir=cache_dir)

        repo = os.path.join(cache_dir, "https", "huggingface.co", "onnx-community")
        assert [c.args[1] for c in mock_download.call_args_list] == [
            os.path.join(repo, "small", "resolve", "main", "onnx", "model.onnx"),
            os.path.join(repo, "small", "resolve", "main", "onnx", "model.onnx_data"),
            os.path.join(repo, "large", "resolve", "main", "onnx", "model.onnx"),
            os.path.join(repo, "large", "resolve", "main", "onnx", "model.onnx_data"),
        ]

    @pytest.mark.parametrize("cache_dir", [None, "mine"])
    def test_urls_that_differ_in_their_query_or_scheme_are_cached_apart(self, cache_dir, monkeypatch, tmp_path):
        # a query can name the file (?id=, ?ref=), and a file fetched over http must never be served for https
        import json
        from unittest import mock

        from kornia.config import kornia_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(kornia_config, "hub_cache_dir", str(tmp_path / "hub"))
        urls = [
            "https://a.invalid/uc?export=download&id=AAA",
            "https://a.invalid/uc?export=download&id=BBB",
            "http://a.invalid/uc?export=download&id=AAA",
            "https://a.invalid/uc",
        ]

        def serve(url, dst, **kwargs):
            Path(dst).write_text(json.dumps({"url": url}))

        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
            assert [ONNXLoader.load_config(url, **kwargs)["url"] for url in urls] == urls
            assert [ONNXLoader.load_config(url, **kwargs)["url"] for url in urls] == urls  # served from the cache

    def test_urls_that_spell_out_a_cache_directory_are_cached_apart(self, monkeypatch, tmp_path):
        # the query's directory and the port are added to the name, and a URL whose own directory or host is spelt
        # like them must not be served the other's file: "q-<hash>" and "<host>_<port>" used to be such names
        import json
        from unittest import mock

        from kornia.config import kornia_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(kornia_config, "hub_cache_dir", str(tmp_path / "hub"))
        q = hashlib.sha256(b"revision=trusted").hexdigest()[:16]
        urls = [
            "https://models.invalid/download/model.json?revision=trusted",
            f"https://models.invalid/download/q-{q}/model.json",
            f"https://models.invalid/download/%3F{q}/model.json",
            "https://a.invalid:8080/model.json",
            "https://a.invalid_8080/model.json",
            "https://a.invalid%3A8080/model.json",
        ]

        def serve(url, dst, **kwargs):
            Path(dst).write_text(json.dumps({"url": url}))

        with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
            assert [ONNXLoader.load_config(url)["url"] for url in urls] == urls
            assert [ONNXLoader.load_config(url)["url"] for url in reversed(urls)] == urls[::-1]  # from the cache

    @pytest.mark.parametrize("url", ["https:///m.onnx", "https:///a/m.onnx", "https://user@/a/m.onnx"])
    def test_load_model_url_without_a_host_raises(self, url, monkeypatch, tmp_path):
        # without its host, "https:///a/m.onnx" would be cached as "https://a/m.onnx", which urlopen fetches while it
        # refuses the former
        from unittest import mock

        monkeypatch.chdir(tmp_path)
        with (
            mock.patch.object(ONNXLoader, "download") as mock_download,
            mock.patch("onnx.load"),
            pytest.raises(ValueError, match="no host"),
        ):
            ONNXLoader.load_model(url)
        mock_download.assert_not_called()

    def test_load_config_with_an_empty_cache_dir_uses_the_default_directory(self, monkeypatch, tmp_path):
        # as for load_model: "" is the default directory, not the working directory
        import json
        from unittest import mock

        from kornia.config import kornia_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(kornia_config, "hub_cache_dir", str(tmp_path / "hub"))

        def serve(url, dst, **kwargs):
            Path(dst).write_text(json.dumps({"url": url}))

        with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
            ONNXLoader.load_config("https://a.invalid/x/cfg.json", cache_dir="")

        assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")] == [
            "hub/https/a.invalid/x/cfg.json"
        ]

    def test_load_config_caches_each_url_separately(self, monkeypatch, tmp_path):
        # #5902: every config was cached as <hub_cache_dir>/<last URL component>, so the first one downloaded was
        # served for every other URL ending in the same file name
        import json
        from unittest import mock

        from kornia.config import kornia_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(kornia_config, "hub_cache_dir", str(tmp_path / "hub"))
        urls = [
            "https://a.invalid/preprocessor_config.json",
            "https://a.invalid/other/preprocessor_config.json",
            "https://b.invalid/preprocessor_config.json",
            "https://b.invalid/cfg.json?x=1",
        ]

        def serve(url, dst, **kwargs):
            Path(dst).write_text(json.dumps({"url": url}))

        with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
            assert [ONNXLoader.load_config(url)["url"] for url in urls] == urls
            assert [ONNXLoader.load_config(url)["url"] for url in urls] == urls  # served from the cache

        q = "%3F" + hashlib.sha256(b"x=1").hexdigest()[:16]
        assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")) == [
            "hub/https/a.invalid/other/preprocessor_config.json",
            "hub/https/a.invalid/preprocessor_config.json",
            f"hub/https/b.invalid/{q}/cfg.json",
            "hub/https/b.invalid/preprocessor_config.json",
        ]

    @pytest.mark.parametrize("form", ["str", "Path", "~"])
    @pytest.mark.parametrize("download", [True, False])
    def test_load_config_reads_a_local_file(self, download, form, monkeypatch, tmp_path):
        # #6119: an existing local file was reported "not found", with "You may set `download=True`"
        import json
        from unittest import mock

        # UTF-8 with a non-ASCII value: a text-mode read decodes it by the locale (cp1252 on Windows) and mangles it.
        expected = {"do_resize": True, "size": {"height": 518}, "processor_class": "R\u00e9sum\u00e9Processor"}
        config = tmp_path / "preprocessor_config.json"
        config.write_bytes(json.dumps(expected, ensure_ascii=False).encode("utf-8"))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what os.path.expanduser reads on Windows
        url = {
            "str": str(config),
            "Path": config,
            "~": os.path.join("~", "preprocessor_config.json"),
        }[form]

        with mock.patch("kornia.core.download._download_url_to_file") as mock_download:
            assert ONNXLoader.load_config(url, download=download) == expected
        mock_download.assert_not_called()

    @pytest.mark.parametrize("what", ["missing file", "directory"])
    @pytest.mark.parametrize("download", [True, False])
    def test_load_config_names_what_it_accepts(self, download, what, tmp_path):
        url = str(tmp_path / "missing.json") if what == "missing file" else str(tmp_path)
        with pytest.raises(RuntimeError, match=r"neither an HTTP\(S\) URL nor an existing local file") as info:
            ONNXLoader.load_config(url, download=download)
        assert "download=True" not in str(info.value)

    def test_path_like_names_are_local_paths(self, monkeypatch, tmp_path):
        # a PathLike names a local file, never an hf:// model or a URL, whatever its text
        from unittest import mock

        monkeypatch.chdir(tmp_path)  # a default cache directory, if one were reached, is relative to it

        class _Name:
            def __init__(self, text):
                self.text = text

            def __fspath__(self):
                return self.text

        with (
            mock.patch.object(ONNXLoader, "download") as mock_download,
            mock.patch("kornia.core.download._download_url_to_file") as mock_fetch,
        ):
            with pytest.raises(ValueError, match=r"File hf://operators/some_model not found"):
                ONNXLoader.load_model(_Name("hf://operators/some_model"))
            with pytest.raises(ValueError, match=r"File https://example\.invalid/m\.onnx not found"):
                ONNXLoader.load_model(_Name("https://example.invalid/m.onnx"))
            with pytest.raises(RuntimeError, match="neither an HTTP"):
                ONNXLoader.load_config(_Name("https://example.invalid/cfg.json"))
        mock_download.assert_not_called()
        mock_fetch.assert_not_called()

    @pytest.mark.parametrize("form", ["Path", "~"])
    def test_load_model_reads_a_local_path_object_and_a_tilde(self, form, monkeypatch, tmp_path):
        from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

        graph = make_graph(
            [make_node("Identity", ["input"], ["output"])],
            "identity",
            [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
        )
        path = tmp_path / "identity.onnx"
        onnx.save(make_model(graph), str(path))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        name = {"Path": path, "~": os.path.join("~", "identity.onnx")}[form]

        assert [node.op_type for node in ONNXLoader.load_model(name).graph.node] == ["Identity"]

    @pytest.mark.parametrize("value", [b"model.onnx", 1, None])
    def test_loaders_take_a_str_or_a_path_object_only(self, value):
        # bytes too: ONNX Runtime reads them as a serialized model, and the message says how to load one
        hint = (
            r" For a serialized model, pass onnx\.load_model_from_string\(data\)\." if isinstance(value, bytes) else ""
        )
        message = rf" takes a str or an os\.PathLike, got {type(value).__name__}\.{hint}$"
        with pytest.raises(TypeError, match=r"ONNXLoader\.load_model" + message):
            ONNXLoader.load_model(value)
        with pytest.raises(TypeError, match=r"ONNXLoader\.load_config" + message):
            ONNXLoader.load_config(value)

    @pytest.mark.parametrize(
        "url", ["https://example.invalid", "https://example.invalid/", "https://example.invalid/d/"]
    )
    def test_load_model_url_without_a_file_name_raises(self, url, monkeypatch, tmp_path):
        from unittest import mock

        monkeypatch.chdir(tmp_path)
        with (
            mock.patch.object(ONNXLoader, "download") as mock_download,
            mock.patch("onnx.load"),
            pytest.raises(ValueError, match="file name"),
        ):
            ONNXLoader.load_model(url)
        mock_download.assert_not_called()

    def test_load_model_hf_name_outside_the_hub_directory_raises(self, monkeypatch, tmp_path):
        # the folder of an hf:// name is joined into the default directory, so it is checked as the rest is
        from unittest import mock

        monkeypatch.chdir(tmp_path)
        with mock.patch.object(ONNXLoader, "download") as mock_download, pytest.raises(ValueError, match="relative"):
            ONNXLoader.load_model("hf://../x")
        mock_download.assert_not_called()

    def test_load_model_rejects_an_unknown_keyword(self, monkeypatch, tmp_path):
        # #5903: load_model forwards its keyword arguments to download_to_cache, which used to drop a misspelt one
        from unittest import mock

        monkeypatch.chdir(tmp_path)
        with (
            mock.patch.object(ONNXLoader, "download") as mock_download,
            mock.patch("onnx.load"),
            pytest.raises(TypeError, match="cachedir"),
        ):
            ONNXLoader.load_model("hf://operators/some_model", cachedir="mine")
        mock_download.assert_not_called()

    def test_unknown_keyword_raises_for_every_kind_of_name(self, tmp_path):
        # a local model or a config URL that is not http(s) never reaches download_to_cache, so the keywords are
        # checked on entry: a misspelt cache_dir must not pass silently on one branch only
        from unittest import mock

        local = tmp_path / "local.onnx"
        local.write_bytes(b"")
        with mock.patch("onnx.load") as mock_onnx_load:
            with pytest.raises(TypeError, match="cachedir"):
                ONNXLoader.load_model(str(local), cachedir="mine")
            with pytest.raises(TypeError, match="cachedir"):
                ONNXLoader.load_model("https://example.invalid/m.onnx", download=False, cachedir="mine")
        mock_onnx_load.assert_not_called()
        with pytest.raises(TypeError, match="cachedir"):
            ONNXLoader.load_config(str(tmp_path / "cfg.json"), cachedir="mine")

    def test_load_model_not_found(self):
        model_name = "non_existent_model.onnx"
        with pytest.raises(ValueError, match=f"File {model_name} not found"):
            ONNXLoader.load_model(model_name)

    def test_download_success(self, tmp_path):
        from unittest import mock

        with mock.patch(
            "kornia.core.download._download_url_to_file",
            side_effect=lambda url, path, **kwargs: Path(path).write_bytes(b"model"),
        ) as mock_download:
            url = "https://huggingface.co/some_model.onnx"
            file_path = tmp_path / "cache" / "some_model.onnx"

            ONNXLoader.download(url, str(file_path))

            mock_download.assert_called_once()
            assert mock_download.call_args.args[0] == url
            assert file_path.read_bytes() == b"model"

    def test_download_failure(self, tmp_path):
        from unittest import mock

        with mock.patch(
            "kornia.core.download._download_url_to_file",
            side_effect=urllib.error.HTTPError(url=None, code=404, msg="Not Found", hdrs=None, fp=None),
        ) as _:
            url = "https://huggingface.co/non_existent_model.onnx"
            file_path = str(tmp_path / "non_existent_model.onnx")

            with pytest.raises(ValueError, match="Error in resolving"):
                ONNXLoader.download(url, file_path)

    def test_fetch_repo_contents_success(self):
        import json
        import os
        from unittest import mock

        with mock.patch("kornia.onnx.utils.urlopen") as mock_urlopen:
            mock_response = mock.MagicMock()
            mock_response.read.return_value = json.dumps([{"path": os.path.join("operators", "model.onnx")}]).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_response

            contents = ONNXLoader._fetch_repo_contents("operators")
            assert contents == [{"path": os.path.join("operators", "model.onnx")}]

    def test_fetch_repo_contents_failure(self):
        from unittest import mock

        url = "https://huggingface.co/api/models/kornia/ONNX_models/tree/main/operators"
        with (
            mock.patch(
                "kornia.onnx.utils.urlopen",
                side_effect=urllib.error.HTTPError(url, 404, "Not Found", {}, None),
            ),
            pytest.raises(ValueError, match="Failed to fetch repository contents"),
        ):
            ONNXLoader._fetch_repo_contents("operators")

    def test_list_operators(self, capsys):
        import os
        from unittest import mock

        with mock.patch("kornia.onnx.utils.ONNXLoader._fetch_repo_contents") as mock_fetch_repo_contents:
            mock_fetch_repo_contents.return_value = [{"path": os.path.join("operators", "some_model.onnx")}]

            ONNXLoader.list_operators()

            captured = capsys.readouterr()
            assert (
                os.path.join("operators", "some_model.onnx").replace("\\", "\\\\") in captured.out
            )  # .replace() for Windows

    def test_list_models(self, capsys):
        import os
        from unittest import mock

        with mock.patch("kornia.onnx.utils.ONNXLoader._fetch_repo_contents") as mock_fetch_repo_contents:
            mock_fetch_repo_contents.return_value = [{"path": os.path.join("operators", "some_model.onnx")}]

            ONNXLoader.list_models()

            captured = capsys.readouterr()
            assert (
                os.path.join("operators", "some_model.onnx").replace("\\", "\\\\") in captured.out
            )  # .replace() for Windows


def test_io_name_conversion():
    from unittest import mock

    from kornia.onnx.utils import io_name_conversion

    with mock.patch("kornia.core.external.onnx.ModelProto") as mock_model_proto:
        # Arrange
        mock_model = mock_model_proto()
        mock_in_node = mock.Mock()
        mock_in_node.name = "input_1"
        mock_out_node = mock.Mock()
        mock_out_node.name = "output_1"
        mock_model.graph.input = [mock_in_node]
        mock_model.graph.output = [mock_out_node]

        mock_mid_node = mock.Mock()
        mock_mid_node.input = ["input_1"]
        mock_mid_node.output = ["output_1"]
        mock_model.graph.node = [mock_mid_node]

        mapping = {"input_1": "input", "output_1": "output"}

        # Act
        converted_model = io_name_conversion(mock_model, mapping)

        # Assert
        assert converted_model.graph.input[0].name == "input"
        assert converted_model.graph.output[0].name == "output"
        assert converted_model.graph.node[0].input[0] == "input"
        assert converted_model.graph.node[0].output[0] == "output"


def test_io_name_conversion_on_a_real_model_runs_in_onnxruntime():
    # the names a model from onnx-community carries, renamed to the ones ONNXSequential connects by default
    import numpy as np

    ort = pytest.importorskip("onnxruntime")
    from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info

    from kornia.onnx import ONNXSequential
    from kornia.onnx.utils import io_name_conversion

    def _vector(name):
        return make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2])

    graph = make_graph(
        [make_node("Neg", ["pixel_values"], ["mid"]), make_node("Relu", ["mid"], ["predicted_depth"])],
        "g",
        [_vector("pixel_values")],
        [_vector("predicted_depth")],
    )
    model = make_model(graph, opset_imports=[make_opsetid("", 18)], ir_version=9)

    converted = io_name_conversion(model, {"pixel_values": "input", "predicted_depth": "output", "absent": "x"})

    assert converted is model  # in place
    assert [v.name for v in model.graph.input] == ["input"]
    assert [v.name for v in model.graph.output] == ["output"]
    assert [list(node.input) for node in model.graph.node] == [["input"], ["mid"]]
    onnx.checker.check_model(model)
    x = np.array([-1.0, 2.0], dtype=np.float32)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    assert session.run(None, {"input": x})[0].tolist() == [1.0, 0.0]
    assert ONNXSequential(model, model)(x)[0].tolist() == [0.0, 0.0]


def test_add_metadata():
    from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

    import kornia
    from kornia.onnx.utils import add_metadata

    graph = make_graph(
        [make_node("Identity", ["input"], ["output"])],
        "identity",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    model = add_metadata(make_model(graph), [("test_key", "test_value")])
    assert [(p.key, p.value) for p in model.metadata_props] == [
        ("source", "kornia"),
        ("version", kornia.__version__),
        ("test_key", "test_value"),
    ]

    # A second call overwrites the existing keys instead of appending duplicates, which check_model rejects.
    model = add_metadata(model, [("test_key", 2)])
    assert [(p.key, p.value) for p in model.metadata_props] == [
        ("source", "kornia"),
        ("version", kornia.__version__),
        ("test_key", "2"),
    ]
    onnx.checker.check_model(model)


def test_add_metadata_merges_duplicate_keys_already_in_the_model():
    from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

    import kornia
    from kornia.onnx.utils import add_metadata

    graph = make_graph(
        [make_node("Identity", ["input"], ["output"])],
        "identity",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    model = make_model(graph)
    # Earlier versions appended on every call, so a model exported and then tagged again repeats its keys. A key the
    # call does not set keeps its last value, the one onnxruntime reads.
    for key, value in [
        ("source", "kornia"),
        ("version", "0.8.0"),
        ("author", "a"),
        ("source", "kornia"),
        ("version", "0.8.0"),
        ("author", "b"),
    ]:
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value

    model = add_metadata(model, [("date", "20261001")])
    assert [(p.key, p.value) for p in model.metadata_props] == [
        ("source", "kornia"),
        ("version", kornia.__version__),
        ("author", "b"),
        ("date", "20261001"),
    ]
    onnx.checker.check_model(model)
