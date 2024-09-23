import os
import urllib
from unittest import mock

import pytest
from onnx import ModelProto  # Assuming `onnx` is installed and ModelProto is part of the library

from kornia.onnx.utils import ONNXLoader


class TestONNXLoader:

    def test_get_file_path_with_custom_cache_dir(self,):
        model_name = os.path.join("operators", "some_model")
        expected_path = os.path.join(".test_cache", "operators", "some_model.onnx")
        assert ONNXLoader._get_file_path(model_name, ".test_cache", suffix=".onnx") == expected_path

    def test_get_file_path_with_default_cache_dir(self):
        model_name = os.path.join("onnx_models", "some_model")
        expected_path = os.path.join(".kornia_hub", "onnx_models", "some_model.onnx")
        assert ONNXLoader._get_file_path(model_name, None, suffix=".onnx") == expected_path

    @mock.patch("onnx.load")
    @mock.patch("os.path.exists")
    def test_load_model_local(self, mock_exists, mock_onnx_load):
        model_name = "local_model.onnx"
        mock_exists.return_value = True

        # Simulate onnx.load returning a dummy ModelProto
        mock_model = mock.Mock(spec=ModelProto)
        mock_onnx_load.return_value = mock_model

        model = ONNXLoader.load_model(model_name)
        assert model == mock_model
        mock_onnx_load.assert_called_once_with(model_name)

    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("os.path.exists")
    def test_load_model_download(self, mock_exists, mock_urlretrieve):
        model_name = "hf://operators/some_model"
        mock_exists.return_value = False
        mock_urlretrieve.return_value = None  # Simulating successful download

        with mock.patch("onnx.load") as mock_onnx_load:
            mock_model = mock.Mock(spec=ModelProto)
            mock_onnx_load.return_value = mock_model

            model = ONNXLoader.load_model(model_name)
            assert model == mock_model
            mock_urlretrieve.assert_called_once_with(
                "https://huggingface.co/kornia/ONNX_models/resolve/main/operators/some_model.onnx",
                os.path.join(".kornia_hub", "onnx_models", "some_model.onnx"),
            )

    def test_load_model_file_not_found(self):
        model_name = "non_existent_model.onnx"

        with pytest.raises(ValueError, match=f"File {model_name} not found"):
            ONNXLoader.load_model(model_name)

    @mock.patch("urllib.request.urlretrieve")
    @mock.patch("os.makedirs")
    def test_download_success(self, mock_makedirs, mock_urlretrieve):
        url = "https://huggingface.co/some_model.onnx"
        file_path = os.path.join(".test_cache", "some_model.onnx")

        ONNXLoader.download(url, file_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(file_path), exist_ok=True)
        mock_urlretrieve.assert_called_once_with(url, file_path)

    @mock.patch(
        "urllib.request.urlretrieve",
        side_effect=urllib.error.HTTPError(url=None, code=404, msg="Not Found", hdrs=None, fp=None),
    )
    def test_download_failure(self, mock_urlretrieve):
        url = "https://huggingface.co/non_existent_model.onnx"
        file_path = os.path.join(".test_cache", "non_existent_model.onnx")

        with pytest.raises(ValueError, match="Error in resolving"):
            ONNXLoader.download(url, file_path)

    @mock.patch("requests.get")
    def test_fetch_repo_contents_success(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"path": os.path.join("operators", "model.onnx")}]
        mock_get.return_value = mock_response

        contents = ONNXLoader._fetch_repo_contents("operators")
        assert contents == [{"path": os.path.join("operators", "model.onnx")}]

    @mock.patch("requests.get")
    def test_fetch_repo_contents_failure(self, mock_get):
        mock_response = mock.Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to fetch repository contents"):
            ONNXLoader._fetch_repo_contents("operators")

    @mock.patch("kornia.onnx.utils.ONNXLoader._fetch_repo_contents")
    def test_list_operators(self, mock_fetch_repo_contents, capsys):
        mock_fetch_repo_contents.return_value = [{"path": os.path.join("operators", "some_model.onnx")}]

        ONNXLoader.list_operators()

        captured = capsys.readouterr()
        assert (
            os.path.join("operators", "some_model.onnx").replace("\\", "\\\\") in captured.out
        )  # .replace() for Windows

    @mock.patch("kornia.onnx.utils.ONNXLoader._fetch_repo_contents")
    def test_list_models(self, mock_fetch_repo_contents, capsys):
        mock_fetch_repo_contents.return_value = [{"path": os.path.join("operators", "some_model.onnx")}]

        ONNXLoader.list_models()

        captured = capsys.readouterr()
        assert (
            os.path.join("operators", "some_model.onnx").replace("\\", "\\\\") in captured.out
        )  # .replace() for Windows
