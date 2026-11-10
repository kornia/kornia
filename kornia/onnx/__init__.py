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

"""Run ONNX models with ONNX Runtime, alone or chained into one graph.

Every runner takes models as ``onnx.ModelProto`` objects, local file paths, HTTP(S) URLs or ``hf://<folder>/<name>``
names of the `kornia/ONNX_models <https://huggingface.co/kornia/ONNX_models>`_ Hugging Face repository, and by default
caches a download in ``kornia.config.kornia_config.hub_onnx_dir`` (``.kornia_hub/onnx_models`` under the working
directory). A runner is called on numpy arrays or CPU torch tensors and returns a list of numpy arrays; its session
runs on the CPU unless ``providers`` or an ``as_*()`` method such as ``as_cuda()`` select another execution provider.
Export a Kornia module to ONNX with ``torch.onnx.export``.

Which function or class to use:

- Run one model: :func:`load`, or :class:`~kornia.onnx.module.ONNXModule` to choose execution providers, session
  options, the cache directory or an IR/opset conversion.
- Run several models one after another as one merged graph: :class:`~kornia.onnx.sequential.ONNXSequential`.
- Load a model as an ``onnx.ModelProto`` or a JSON config as a ``dict``, without a session, or list the Hub
  repository: :class:`~kornia.onnx.utils.ONNXLoader`.
- Edit a model in place: its metadata, :func:`~kornia.onnx.utils.add_metadata`; its input and output names,
  :func:`~kornia.onnx.utils.io_name_conversion`.
- Cache any HTTP(S) file under a given name: :class:`~kornia.onnx.download.CachedDownloader`.
"""

from .module import *
from .sequential import *
from .utils import *
