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

:func:`load`, ``ONNXModule`` and ``ONNXSequential`` take models as ``onnx.ModelProto`` objects, local file paths,
HTTP(S) URLs or ``hf://<folder>/<name>`` names of the `kornia/ONNX_models <https://huggingface.co/kornia/ONNX_models>`_
Hugging Face repository, and by default cache a download in ``kornia.config.kornia_config.hub_onnx_dir``
(``.kornia_hub/onnx_models`` under the working directory). They are called on numpy arrays or CPU torch tensors and
return a list of numpy arrays; the session runs on the CPU unless ``providers`` or an ``as_*()`` method such as
``as_cuda()`` selects another execution provider.
Export a Kornia module with ``torch.onnx.export``, or with its ``to_onnx()`` method where it has one, which returns
the ``onnx.ModelProto`` these classes take.

Which function or class to use:

- Run one model: :func:`load`, or :class:`~kornia.onnx.module.ONNXModule` to choose execution providers, session
  options, the cache directory or an IR/opset conversion.
- Run several models one after another as one merged graph: :class:`~kornia.onnx.sequential.ONNXSequential`.
- Load a model as an ``onnx.ModelProto`` or a JSON config as a ``dict``, without a session, or list the Hub
  repository: :class:`~kornia.onnx.utils.ONNXLoader`.
- Add ``source``, ``version`` and other metadata entries to a model in place: :func:`~kornia.onnx.utils.add_metadata`.
- Rename the inputs and outputs of a model in place: :func:`~kornia.onnx.utils.io_name_conversion`.
- Cache any HTTP(S) file under a given name: :class:`~kornia.onnx.download.CachedDownloader`.
"""

from .module import *
from .sequential import *
from .utils import *
