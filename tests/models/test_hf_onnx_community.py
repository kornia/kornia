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

from onnx import TensorProto, helper  # noqa: E402

from kornia.models._hf_models.hf_onnx_community import HFONNXComunnityModelLoader  # noqa: E402


def test_add_metadata_replaces_duplicate_keys_and_is_idempotent():
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.metadata_props.add(key="source", value="huggingface")
    model.metadata_props.add(key="keep", value="yes")
    model.metadata_props.add(key="source", value="duplicate")
    loader = HFONNXComunnityModelLoader("test", cache_dir=".")
    metadata = [("source", "kornia"), ("version", "1")]

    loader._add_metadata(model, metadata)
    loader._add_metadata(model, metadata)

    assert [(prop.key, prop.value) for prop in model.metadata_props] == [
        ("source", "kornia"),
        ("keep", "yes"),
        ("version", "1"),
    ]
    onnx.checker.check_model(model)


def test_add_metadata_accepts_a_dict():
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    loader = HFONNXComunnityModelLoader("test", cache_dir=".")

    loader._add_metadata(model, {"version": 1})

    assert [(prop.key, prop.value) for prop in model.metadata_props] == [("version", "1")]
    onnx.checker.check_model(model)
