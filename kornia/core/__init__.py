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

"""Fetch weights, subclass the image and ONNX base classes, and build small batched tensors.

The download functions try a list of URLs in order and retry transient failures. The device and autocast helpers and
:func:`~kornia.core.utils.batched_forward` are imported from ``kornia.core.utils``; the ``KORNIA_CHECK*`` input checks
from ``kornia.core.check``, and the errors they raise from ``kornia.core.exceptions``.

Which function to use:

- A checkpoint :func:`torch.load` reads, from one URL or a list of fallback URLs: :func:`load_state_dict_from_url`.
- A file downloaded into the cache and returned as a path, without loading it: :func:`download_file_from_url`; one
  file of a Hugging Face repository: :func:`download_hf_file`; only that file's URL: :func:`hf_url`.
- A ``.safetensors`` checkpoint read into a state dict: :func:`load_safetensors`; its header checked without
  reading tensors, as the ``validate`` of a download: :func:`check_safetensors`.
- An ``nn.Module`` whose call also takes NumPy arrays, PIL images and image paths and returns tensors, arrays or PIL
  images: subclass :class:`ImageModule`, or chain modules in :class:`ImageSequential`; the conversions alone, for
  any function: :class:`ImageModuleMixIn`.
- ONNX: a ``to_onnx`` export method for a module, :class:`ONNXExportMixin`; an ONNX Runtime session with a call on
  arrays and tensors, :class:`ONNXRuntimeMixin`; loading and chaining ONNX graphs, :class:`ONNXMixin`.
- Batched identity matrices or zero vectors with another tensor's batch size, dtype and device: :func:`eye_like`,
  :func:`vec_like`.
- A tensor wrapper whose torch functions and operators return the wrapper's class: :class:`TensorWrapper`.
- The lazily imported optional dependencies (NumPy, Pillow, ``onnx``, ``onnxruntime``, ``diffusers``): ``external``.
"""

from . import external
from .download import download_file_from_url, download_hf_file, hf_url, load_state_dict_from_url
from .mixin import (
    ImageModuleMixIn,
    ONNXExportMixin,
    ONNXMixin,
    ONNXRuntimeMixin,
)
from .module import ImageModule, ImageSequential
from .ops import eye_like, vec_like
from .safetensors import check_safetensors, load_safetensors
from .tensor_wrapper import TensorWrapper

__all__ = [
    "ImageModule",
    "ImageModuleMixIn",
    "ImageSequential",
    "ONNXExportMixin",
    "ONNXMixin",
    "ONNXRuntimeMixin",
    "TensorWrapper",
    "check_safetensors",
    "download_file_from_url",
    "download_hf_file",
    "external",
    "eye_like",
    "hf_url",
    "load_safetensors",
    "load_state_dict_from_url",
    "vec_like",
]
