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

pytestmark = pytest.mark.device_agnostic

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

import kornia  # noqa: E402
from kornia.models._hf_models.hf_onnx_community import HFONNXComunnityModel  # noqa: E402


def _identity_model(*metadata: tuple[str, str]) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    for key, value in metadata:
        model.metadata_props.add(key=key, value=value)
    return model


@pytest.mark.parametrize("include_pre_and_post_processor", [False, True])
def test_to_onnx_keeps_one_metadata_entry_per_key(include_pre_and_post_processor):
    # #5263: a key the downloaded model already carries and a second export must each leave one entry per key, since
    # onnx.checker.check_model rejects duplicate keys.
    model = _identity_model(("source", "huggingface"), ("keep", "yes"))
    hf_model = HFONNXComunnityModel(model, pre_processor=_identity_model())

    hf_model.to_onnx(
        save=False,
        include_pre_and_post_processor=include_pre_and_post_processor,
        additional_metadata=[("date", "1")],
    )
    exported = hf_model.to_onnx(
        save=False,
        include_pre_and_post_processor=include_pre_and_post_processor,
        additional_metadata=[("date", "2")],
    )

    assert [(prop.key, prop.value) for prop in exported.metadata_props] == [
        ("source", "kornia"),
        ("keep", "yes"),
        ("version", kornia.__version__),
        ("date", "2"),
    ]
    onnx.checker.check_model(exported)


def test_model_with_its_own_source_chains_with_a_kornia_pre_processor():
    # #6117: the model's "source" differs from the one kornia writes into the pre-processor, which merge_models
    # refused: "Can't merge models with different values for the same model metadata property".
    import numpy as np

    model = _identity_model(("source", "optimum"))
    pre_processor = _identity_model(("source", "kornia"), ("version", kornia.__version__))

    hf_model = HFONNXComunnityModel(model, pre_processor=pre_processor)

    assert hf_model(np.array([3.0], dtype=np.float32))[0].tolist() == [3.0]
    # the later model's value wins; the caller's graphs keep their own
    assert [(prop.key, prop.value) for prop in hf_model._combined_op.metadata_props] == [
        ("source", "optimum"),
        ("version", kornia.__version__),
    ]
    assert [(prop.key, prop.value) for prop in model.metadata_props] == [("source", "optimum")]


@pytest.mark.parametrize("include_pre_and_post_processor", [False, True])
def test_to_onnx_leaves_the_stored_and_the_callers_graphs_unchanged(include_pre_and_post_processor):
    # Each export carries that call's metadata on a copy: the keys of one call used to be written into the stored
    # graph, which is the caller's ModelProto, and so into every later export.
    model = _identity_model(("keep", "yes"))
    hf_model = HFONNXComunnityModel(model, pre_processor=_identity_model())
    stored = hf_model._combined_op.SerializeToString()

    first = hf_model.to_onnx(
        save=False, include_pre_and_post_processor=include_pre_and_post_processor, additional_metadata=[("run", "1")]
    )
    second = hf_model.to_onnx(save=False, include_pre_and_post_processor=include_pre_and_post_processor)

    assert ("run", "1") in [(prop.key, prop.value) for prop in first.metadata_props]
    assert "run" not in [prop.key for prop in second.metadata_props]
    assert [(prop.key, prop.value) for prop in model.metadata_props] == [("keep", "yes")]
    assert hf_model.model is model
    assert hf_model._combined_op.SerializeToString() == stored


def test_resize_back_names_the_expected_type():
    import torch

    hf_model = HFONNXComunnityModel(_identity_model())
    with pytest.raises(TypeError, match=r"target_images must be a torch\.Tensor, got list"):
        hf_model.resize_back(torch.rand(1, 3, 4, 4), [torch.rand(3, 8, 8)])


def test_each_repository_preprocessing_config_is_cached_beside_its_model(monkeypatch, tmp_path):
    # #5902: every onnx-community repository's preprocessor_config.json used to be cached as one file,
    # <hub_cache_dir>/preprocessor_config.json, outside the loader's cache_dir, so the second model built loaded the
    # first one's preprocessing
    import json
    from pathlib import Path
    from unittest import mock

    import torch

    from kornia.config import kornia_config
    from kornia.models._hf_models import HFONNXComunnityModelLoader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(kornia_config, "hub_cache_dir", str(tmp_path / "hub"))
    sizes = {"depth-anything-v2-small": 14, "depth-anything-v2-large": 28}

    def serve(url, dst, **kwargs):
        (size,) = (size for name, size in sizes.items() if f"/onnx-community/{name}/" in url)
        config = {
            "image_processor_type": "DPTImageProcessor",
            "do_pad": False,
            "do_resize": True,
            "size": {"height": size, "width": size},
            "do_rescale": False,
            "do_normalize": False,
        }
        Path(dst).write_text(json.dumps(config))

    # one root, as DepthAnythingBuilder.build(cache_dir=None) uses for every repository
    loaders = {name: HFONNXComunnityModelLoader(name, cache_dir=str(tmp_path / "hub")) for name in sizes}
    image = torch.rand(1, 3, 7, 9)
    with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
        for name, loader in loaders.items():
            preprocessing = loader.load_preprocessing()
            assert preprocessing(image).shape[-2:] == (sizes[name], sizes[name]), name

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")) == [
        "hub/depth-anything-v2-large/preprocessor_config.json",
        "hub/depth-anything-v2-small/preprocessor_config.json",
    ]


@pytest.mark.parametrize("with_data", [False, True])
def test_loader_keeps_the_model_cache_paths_of_earlier_releases(with_data, tmp_path):
    # <cache_dir>/<model_name>/<model_type>.onnx(_data): a model of up to 1.3 GB that is already cached there must not
    # be downloaded again
    from unittest import mock

    from kornia.models._hf_models import HFONNXComunnityModelLoader
    from kornia.onnx.utils import ONNXLoader

    loader = HFONNXComunnityModelLoader(
        "depth-anything-v2-large", "model_fp16", cache_dir=str(tmp_path), with_data=with_data
    )
    with mock.patch.object(ONNXLoader, "download") as mock_download, mock.patch("onnx.load"):
        loader.load_model()
    url = "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model_fp16.onnx"
    expected = [(url, str(tmp_path / "depth-anything-v2-large" / "model_fp16.onnx"))]
    if with_data:
        expected.append((url + "_data", str(tmp_path / "depth-anything-v2-large" / "model_fp16.onnx_data")))
    assert [c.args[:2] for c in mock_download.call_args_list] == expected


@pytest.mark.parametrize("model_name", ["../outside", "a/../../outside", "", ".", "ABSOLUTE"])
def test_loader_refuses_a_model_name_outside_the_cache_directory(model_name, tmp_path):
    # the model, its weights and its preprocessing config are cached in <cache_dir>/<model_name>/: a name that leaves
    # cache_dir would load a file already there, such as <cache_dir>/../outside/preprocessor_config.json, unfetched
    from unittest import mock

    from kornia.models._hf_models import HFONNXComunnityModelLoader
    from kornia.onnx.utils import ONNXLoader

    root = tmp_path / "hub"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "preprocessor_config.json").write_text("{}")
    if model_name == "ABSOLUTE":
        model_name = str(outside)
    with mock.patch.object(ONNXLoader, "download") as mock_download, pytest.raises(ValueError, match="model_name"):
        HFONNXComunnityModelLoader(model_name, cache_dir=str(root)).load_preprocessing()
    mock_download.assert_not_called()


_DPT_CONFIG = {
    "image_processor_type": "DPTImageProcessor",
    "do_pad": False,
    "do_resize": True,
    "size": {"height": 14, "width": 14},
    "do_rescale": False,
    "do_normalize": False,
}


@pytest.fixture
def fake_repository():
    """Serve a repository's model, weights and preprocessing config, and record the file name of every fetch."""
    import json
    from pathlib import Path
    from unittest import mock

    versions = {"model": "1", "config": 1}
    fetched: list[str] = []

    def serve(url, dst, **kwargs):
        name = url.rsplit("/", 1)[-1]
        fetched.append(name)
        if name == "preprocessor_config.json":
            content = json.dumps({**_DPT_CONFIG, "size": {"height": 14 * versions["config"], "width": 14}}).encode()
        elif name.endswith(".onnx_data"):
            content = b"weights"
        else:
            content = _identity_model(("version", versions["model"])).SerializeToString()
        Path(dst).write_bytes(content)

    with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
        yield versions, fetched


def test_loader_force_download_fetches_the_model_weights_and_config_again(fake_repository, tmp_path):
    # a cached file of an onnx-community repository was used for as long as it existed: an updated model was never
    # fetched again
    import torch

    from kornia.models._hf_models import HFONNXComunnityModelLoader

    versions, fetched = fake_repository
    loader = HFONNXComunnityModelLoader("depth-anything-v2-small", cache_dir=str(tmp_path), with_data=True)

    def version(model):
        return {p.key: p.value for p in model.metadata_props}["version"]

    def height(preprocessing):
        return preprocessing(torch.rand(1, 3, 7, 9)).shape[-2]

    assert version(loader.load_model()) == "1"
    assert height(loader.load_preprocessing()) == 14
    assert fetched == ["model.onnx", "model.onnx_data", "preprocessor_config.json"]

    versions["model"], versions["config"] = "2", 2  # the repository changed
    assert version(loader.load_model()) == "1"
    assert height(loader.load_preprocessing()) == 14
    assert len(fetched) == 3  # the cached files are used as they are

    assert version(loader.load_model(force_download=True)) == "2"
    assert fetched[3:] == ["model.onnx", "model.onnx_data"]  # the weights beside the model, as they go together
    assert height(loader.load_preprocessing(force_download=True)) == 28
    assert fetched[5:] == ["preprocessor_config.json"]

    assert version(loader.load_model()) == "2"  # the new files are the cached ones
    assert height(loader.load_preprocessing()) == 28
    assert len(fetched) == 6


def test_loader_damaged_cached_files_name_force_download(fake_repository, tmp_path):
    import json

    from google.protobuf.message import DecodeError

    from kornia.models._hf_models import HFONNXComunnityModelLoader

    _, fetched = fake_repository
    loader = HFONNXComunnityModelLoader("depth-anything-v2-small", cache_dir=str(tmp_path))
    loader.load_model()
    loader.load_preprocessing()
    model, config = (
        tmp_path / "depth-anything-v2-small" / "model.onnx",
        tmp_path / "depth-anything-v2-small" / "preprocessor_config.json",
    )
    model.write_bytes(b"<html>Bad gateway</html>")
    config.write_text("<html>Bad gateway</html>")

    with pytest.raises(DecodeError) as model_error:
        loader.load_model()
    with pytest.raises(json.JSONDecodeError) as config_error:
        loader.load_preprocessing()
    assert "force_download=True" in model_error.value.__notes__[0]
    assert "force_download=True" in config_error.value.__notes__[0]
    assert len(fetched) == 2

    loader.load_model(force_download=True)
    loader.load_preprocessing(force_download=True)
    assert len(fetched) == 4
    loader.load_model()
    loader.load_preprocessing()


def test_loader_force_download_without_download_raises(fake_repository, tmp_path):
    from kornia.models._hf_models import HFONNXComunnityModelLoader

    _, fetched = fake_repository
    loader = HFONNXComunnityModelLoader("depth-anything-v2-small", cache_dir=str(tmp_path))
    loader.load_model()

    with pytest.raises(ValueError, match=r"force_download=True.*download=True"):
        loader.load_model(download=False, force_download=True)
    assert fetched == ["model.onnx"]
