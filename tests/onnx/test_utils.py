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

import pytest

onnx = pytest.importorskip("onnx")

from kornia.onnx.utils import ONNXLoader


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

    def test_load_model_download(self):
        from unittest import mock

        from onnx import ModelProto

        with (
            mock.patch("urllib.request.urlretrieve") as mock_urlretrieve,
            mock.patch("os.path.exists") as mock_exists,
            mock.patch("onnx.load") as mock_onnx_load,
        ):
            model_name = "hf://operators/some_model"
            mock_exists.return_value = False
            mock_urlretrieve.return_value = None  # Simulating successful download

            mock_model = mock.Mock(spec=ModelProto)
            mock_onnx_load.return_value = mock_model

            model = ONNXLoader.load_model(model_name)
            assert model == mock_model
            mock_urlretrieve.assert_called_once_with(
                "https://huggingface.co/kornia/ONNX_models/resolve/main/operators/some_model.onnx",
                os.path.join(".kornia_hub", "onnx_models", "operators", "some_model.onnx"),
            )

    def test_load_model_not_found(self):
        model_name = "non_existent_model.onnx"
        with pytest.raises(ValueError, match=f"File {model_name} not found"):
            ONNXLoader.load_model(model_name)

    def test_download_success(self):
        import os
        from unittest import mock

        with mock.patch("urllib.request.urlretrieve") as mock_urlretrieve, mock.patch("os.makedirs") as mock_makedirs:
            url = "https://huggingface.co/some_model.onnx"
            file_path = os.path.join(".test_cache", "some_model.onnx")

            ONNXLoader.download(url, file_path)

            mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
            mock_urlretrieve.assert_called_once_with(url, file_path)

    def test_download_failure(self):
        import os
        from unittest import mock

        with mock.patch(
            "urllib.request.urlretrieve",
            side_effect=urllib.error.HTTPError(url=None, code=404, msg="Not Found", hdrs=None, fp=None),
        ) as mock_urlretrieve:
            url = "https://huggingface.co/non_existent_model.onnx"
            file_path = os.path.join(".test_cache", "non_existent_model.onnx")

            with pytest.raises(ValueError, match="Error in resolving"):
                ONNXLoader.download(url, file_path)

    def test_fetch_repo_contents_success(self):
        import os
        from unittest import mock

        with mock.patch("requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{"path": os.path.join("operators", "model.onnx")}]
            mock_get.return_value = mock_response

            contents = ONNXLoader._fetch_repo_contents("operators")
            assert contents == [{"path": os.path.join("operators", "model.onnx")}]

    def test_fetch_repo_contents_failure(self):
        from unittest import mock

        with mock.patch("requests.get") as mock_get:
            mock_response = mock.Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            with pytest.raises(ValueError, match="Failed to fetch repository contents"):
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
    from unittest import mock

    from kornia.onnx.utils import add_metadata

    with mock.patch("kornia.core.external.onnx.ModelProto") as mock_model_proto:
        # Arrange
        mock_model = mock_model_proto()
        mock_metadata_props = mock.Mock()
        mock_model.metadata_props.add.return_value = mock_metadata_props

        # Act
        model_with_metadata = add_metadata(mock_model, [("test_key", "test_value")])

        # Assert
        calls = [
            mock.call(),  # for "source"
            mock.call(),  # for "version"
            mock.call(),  # for "test_key"
        ]
        mock_model.metadata_props.add.assert_has_calls(calls)
        assert mock_model.metadata_props.add.call_count == 3
        # Check if version was added
        # (Since it's a mock, we just check if any call set value to kornia.__version__)
        values = [c.value for c in mock_metadata_props.mock_calls if hasattr(c, "value")]
        # Metadata logic: metadata_props.key = key; metadata_props.value = str(value)
