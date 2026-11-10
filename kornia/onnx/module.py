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

"""Run one ONNX model with ONNX Runtime, given as a ``ModelProto``, a file, a URL or an ``hf://`` name.

Which function or class to use:

- One model with the default CPU session and cache: :func:`~kornia.onnx.load`.
- One model with chosen execution providers, session options, cache directory or IR/opset conversion:
  :class:`ONNXModule`.
- Several models run in sequence as one graph: :class:`~kornia.onnx.sequential.ONNXSequential`.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.core.mixin.onnx import ONNXMixin, ONNXRuntimeMixin

__all__ = ["ONNXModule", "load"]


class ONNXModule(ONNXMixin, ONNXRuntimeMixin):
    """Run one ONNX model with ONNX Runtime and return its outputs as numpy arrays.

    ``module = ONNXModule(op)`` loads ``op`` and opens an ONNX Runtime session on it; ``outputs = module(*inputs)``
    takes one numpy array or CPU torch tensor of a dtype numpy has (not bfloat16 or float8) per graph input, in the
    order of ``get_session().get_inputs()``, and returns a list of numpy arrays. The session runs on the CPU unless
    ``providers`` or an ``as_*()`` method such as ``as_cuda()`` select another provider; inputs stay in host memory
    either way, and a CUDA tensor raises ``ValueError``.

    Args:
        op: One ONNX model: an ONNX ModelProto object, or a file path as ``str`` or :class:`os.PathLike` (a path
            object always names a local file). A ``str`` can also be an HTTP(S) URL or an ``hf://<folder>/<name>``
            name of the `kornia/ONNX_models <https://huggingface.co/kornia/ONNX_models>`_ repository, such as
            ``"hf://operators/kornia.color.gray.RgbToGrayscale"``: ``hf://`` and a repository file path without
            ``.onnx``, as :meth:`~kornia.onnx.utils.ONNXLoader.list_operators` and
            :meth:`~kornia.onnx.utils.ONNXLoader.list_models` print the paths.
        providers: A list of execution providers for ONNXRuntime
            (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']). A provider that the installed onnxruntime
            build does not offer is skipped with onnxruntime's ``UserWarning``; the session runs on the CPU if none
            is left. If None, the CPU.
        session_options: ONNX Runtime ``SessionOptions``, used as given. If None, the session uses
            ``ORT_ENABLE_EXTENDED`` graph optimizations.
        cache_dir: The directory where downloaded ONNX models (``hf://`` names and URLs) are cached, as
            :meth:`~kornia.onnx.utils.ONNXLoader.load_model` lays it out. If None,
            ``kornia.config.kornia_config.hub_onnx_dir`` (``.kornia_hub/onnx_models`` under the working directory).
        target_ir_version: The IR version to convert the model to. If None, the model keeps its IR version.
        target_opset_version: The OPSET version to convert the model to. If None, the model keeps its OPSET version.
        force_download: If True, an ``hf://`` name or a URL is downloaded again, and replaces the cached file, even
            when it is cached. By default a cached file is used as it is, however old, and one that does not load
            fails every time until it is deleted. It has no effect on a model object or a path to a local file.

    Raises:
        TypeError: If ``op`` is not a ModelProto, a ``str`` or an :class:`os.PathLike`; ``bytes`` raise it too.
        ValueError: If ``op`` names a local file that does not exist.

    Note:
        Given an ``onnx.ModelProto``, the module holds that object itself, not a copy, unless a target version
        converts it: with no target, or with a target the model already has, ``module.op is op``. Whatever edits the
        wrapped graph in place, such as :meth:`add_metadata`, then changes the caller's model too. Pass a copy
        (``copy.deepcopy(op)``) to keep the original unchanged.

    See Also:
        - :func:`~kornia.onnx.load`: the same module from one argument, with the default CPU session and cache.
        - :class:`~kornia.onnx.sequential.ONNXSequential`: several models merged into one graph and one session.
        - :class:`~kornia.onnx.utils.ONNXLoader`: the ``onnx.ModelProto`` alone, without a session.

    Example:
        >>> import numpy as np
        >>> from onnx import TensorProto, helper
        >>> from kornia.onnx import ONNXModule
        >>> graph = helper.make_graph(
        ...     [helper.make_node("Relu", ["input"], ["output"])], "relu",
        ...     [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])],
        ...     [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3])])
        >>> model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        >>> module = ONNXModule(model)
        >>> module(np.array([[-1.0, 2.0, -3.0]], dtype=np.float32))
        [array([[0., 2., 0.]], dtype=float32)]

    """

    def __init__(
        self,
        op: Union[onnx.ModelProto, str, os.PathLike],
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        cache_dir: Optional[str] = None,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
        *,
        force_download: bool = False,
    ) -> None:
        self.op = self._load_op(op, cache_dir, force_download=force_download)
        if target_ir_version is not None or target_opset_version is not None:
            self.op = self._onnx_version_conversion(
                self.op, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
        session = self.create_session(providers=providers, session_options=session_options)
        self.set_session(session=session)

    def create_session(
        self, providers: list[str] | None = None, session_options: Any | None = None
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime session for the wrapped model graph, without making it the module's session.

        Pass the result to ``set_session`` to run the module on it.

        Args:
            providers: Ordered list of ONNX Runtime execution providers, such as ``"CUDAExecutionProvider"`` followed
                by ``"CPUExecutionProvider"``. A provider the installed onnxruntime build does not offer is skipped
                with onnxruntime's ``UserWarning``. If None, the CPU.
            session_options: ONNX Runtime ``SessionOptions``, used as given. If None, ``ORT_ENABLE_EXTENDED`` graph
                optimizations.

        Returns:
            ONNX Runtime inference session bound to ``self.op``, the wrapped
            ONNX model graph.
        """
        return super()._create_session(self.op, providers, session_options)

    def export(self, file_path: str, **kwargs: Any) -> None:
        """Save the wrapped ONNX model graph to an ``.onnx`` file with ``onnx.save``.

        Args:
            file_path: Destination path for the ``.onnx`` file.
            kwargs: Keyword arguments of ``onnx.save``, such as ``save_as_external_data``.
        """
        return super()._export(self.op, file_path, **kwargs)

    def add_metadata(self, additional_metadata: Optional[list[tuple[str, str]]] = None) -> onnx.ModelProto:
        """Attach metadata entries to the wrapped ONNX model graph.

        The metadata is written into the wrapped graph in place, as :func:`~kornia.onnx.utils.add_metadata` writes
        it. When the module was built from an ``onnx.ModelProto``, the graph is the caller's own object unless a
        target version converted it; see the note of :class:`ONNXModule`. The running session keeps the metadata it
        was created with; :meth:`export` saves the new entries.

        Args:
            additional_metadata: Optional list of ``(key, value)`` string pairs
                to store in the ONNX model metadata properties.

        Returns:
            The wrapped ONNX model graph after metadata has been added.
        """
        return super()._add_metadata(self.op, additional_metadata)


def load(model_name: Union[onnx.ModelProto, str, os.PathLike], *, force_download: bool = False) -> ONNXModule:
    """Load one ONNX model into an :class:`~kornia.onnx.module.ONNXModule` on the CPU, in one call.

    ``load(name, force_download=f)`` is ``ONNXModule(name, force_download=f)``: one model on a CPU session with the
    default cache.

    Args:
        model_name: The model to load, as the ``op`` argument of :class:`~kornia.onnx.module.ONNXModule` takes it: an
            ONNX ModelProto object, a local file path, an HTTP(S) URL or an ``hf://<folder>/<name>`` name.
        force_download: If True, download the model again and replace the cached file, even when it is cached.

    Returns:
        The :class:`~kornia.onnx.module.ONNXModule` on a CPU session.

    Raises:
        TypeError: If ``model_name`` is not a ModelProto, a ``str`` or an :class:`os.PathLike`.
        ValueError: If ``model_name`` names a local file that does not exist.

    Note:
        Given an ``onnx.ModelProto``, the module holds that object itself; see the note of
        :class:`~kornia.onnx.module.ONNXModule`.

    See Also:
        - :class:`~kornia.onnx.module.ONNXModule`: the same module with execution providers, session options, a cache
          directory and version conversion.
        - :meth:`~kornia.onnx.utils.ONNXLoader.load_model`: the ``onnx.ModelProto`` alone, without a session.

    """
    return ONNXModule(model_name, force_download=force_download)
