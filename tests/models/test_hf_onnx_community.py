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

    loaders = {
        "depth-anything-v2-small": HFONNXComunnityModelLoader("depth-anything-v2-small", cache_dir=str(tmp_path / "a")),
        "depth-anything-v2-large": HFONNXComunnityModelLoader("depth-anything-v2-large", cache_dir=str(tmp_path / "b")),
    }
    image = torch.rand(1, 3, 7, 9)
    with mock.patch("kornia.core.download._download_url_to_file", side_effect=serve):
        for name, loader in loaders.items():
            preprocessing = loader.load_preprocessing()
            assert preprocessing(image).shape[-2:] == (sizes[name], sizes[name]), name

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json")) == [
        "a/depth-anything-v2-small/preprocessor_config.json",
        "b/depth-anything-v2-large/preprocessor_config.json",
    ]
