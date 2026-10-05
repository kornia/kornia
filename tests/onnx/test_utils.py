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
import urllib
from pathlib import Path

import pytest

# Every test in this module must stay device-free: the mark deselects the whole file on non-CPU devices.
pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")

from kornia.onnx.utils import ONNXLoader  # noqa: E402


class TestONNXLoader:
    def test_get_file_path(self):
        # Test getting local file path for caching
        model_name = "some_model"
        expected_path = os.path.join(".kornia_hub", "some_model.onnx")
        assert ONNXLoader._get_file_path(model_name, None, suffix=".onnx") == expected_path

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

    def test_load_model_not_found(self):
        model_name = "non_existent_model.onnx"
        with pytest.raises(ValueError, match=f"File {model_name} not found"):
            ONNXLoader.load_model(model_name)

    def test_download_success(self, tmp_path):
        from unittest import mock

        with mock.patch(
            "urllib.request.urlretrieve", side_effect=lambda url, path: Path(path).write_bytes(b"model")
        ) as mock_urlretrieve:
            url = "https://huggingface.co/some_model.onnx"
            file_path = tmp_path / "cache" / "some_model.onnx"

            ONNXLoader.download(url, str(file_path))

            mock_urlretrieve.assert_called_once()
            assert mock_urlretrieve.call_args.args[0] == url
            assert file_path.read_bytes() == b"model"

    def test_download_failure(self, tmp_path):
        from unittest import mock

        with mock.patch(
            "urllib.request.urlretrieve",
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
