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

from __future__ import annotations

import os
from typing import Any, Optional, Union

from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.core.mixin import ONNXMixin, ONNXRuntimeMixin

__all__ = ["ONNXSequential"]


class ONNXSequential(ONNXMixin, ONNXRuntimeMixin):
    """ONNXSequential to chain multiple ONNX operators together.

    Calling the object runs the combined graph on numpy arrays or CPU torch tensors of a dtype numpy has (not
    bfloat16) and returns a list of numpy arrays; see ``ONNXRuntimeMixin.__call__``. The metadata of the models are
    merged into the combined model with one entry per key: each key keeps its first position and takes the value of
    the last model that sets it. The models passed in are not modified.

    Args:
        *args: A variable number of ONNX models (ONNX ModelProto objects, or file paths as ``str``, ``bytes`` or
            :class:`os.PathLike`; a path object always names a local file). For Hugging Face-hosted models, use the
            format 'hf://model_name'. Valid `model_name` can be found on https://huggingface.co/kornia/ONNX_models.
            Or a URL to the ONNX model.
        providers: A list of execution providers for ONNXRuntime
            (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        session_options: Optional ONNXRuntime session options for optimizing the session.
        io_maps: An optional list with one list of ``(output, input)`` name pairs per joint between consecutive
            models: ``io_maps[0]`` connects outputs of the first model to inputs of the second, ``io_maps[1]`` the
            second to the third, and so on, as in ``[[("out_a", "in_b")], [("out_b", "in_c")]]`` for three models.
            Each entry may be any iterable of pairs (a ``zip``, ``dict.items()``, ...), and ``io_maps`` anything
            indexed by the joint; entries past the last joint are not read.
            If None, we assume the default input name and output name are "input" and "output" accordingly, and
            only one input and output node for each graph.
            The merged graph does not check a later model's declared input types: a ``UserWarning`` says so when the
            tensor element types of an output and the input it feeds differ. Declared dimensions are not compared.
        cache_dir: The directory where downloaded ONNX models (``hf://`` names and URLs) are cached locally.
            Defaults to None, which will use a default `kornia.config.hub_onnx_dir` directory.
        auto_ir_version_conversion: If True, convert the models to one IR version and one OPSET version before
            combining them: IR version 9, and the highest OPSET version among the models, to which the others are
            converted up. Other versions may be pointed to by `target_ir_version` and `target_opset_version`.
        target_ir_version: The IR version to convert every model to before combining them, whether or not
            `auto_ir_version_conversion` is set. If None, the models keep their IR versions, unless
            `auto_ir_version_conversion` converts them to 9.
        target_opset_version: The OPSET version to convert every model to before combining them, whether or not
            `auto_ir_version_conversion` is set. If None, the models keep their OPSET versions, unless
            `auto_ir_version_conversion` converts them to the highest among them. The ONNX version converter raises
            where it has no adapter for an operator, for example ``Resize`` or ``Pad`` from opset 18 to 17.

    """

    def __init__(
        self,
        *args: Union[onnx.ModelProto, str, bytes, os.PathLike],
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        io_maps: Optional[list[list[tuple[str, str]]]] = None,
        cache_dir: Optional[str] = None,
        auto_ir_version_conversion: bool = False,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> None:
        self.operators = self._load_ops(*args, cache_dir=cache_dir)
        if auto_ir_version_conversion:
            self.operators = self._auto_version_conversion(
                *self.operators, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
        elif target_ir_version is not None or target_opset_version is not None:
            # as ONNXModule does: only the versions given are changed
            self.operators = [
                self._onnx_version_conversion(
                    op, target_ir_version=target_ir_version, target_opset_version=target_opset_version
                )
                for op in self.operators
            ]
        self._combined_op = self.combine(io_maps=io_maps)
        session = self.create_session(providers=providers, session_options=session_options)
        self.set_session(session=session)

    def _auto_version_conversion(
        self,
        *args: onnx.ModelProto,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
    ) -> list[onnx.ModelProto]:
        """Automatic conversion of the model's IR/OPSET version to the given target version.

        If `target_ir_version` is not provided, the model is converted to 9 by default.
        If `target_opset_version` is not provided, the models are converted to the highest OPSET version among
        them.

        Args:
            args: List of operations to convert.
            target_ir_version: The target IR version to convert to.
            target_opset_version: The target OPSET version to convert to.

        """
        if target_ir_version is None:
            target_ir_version = 9
        if target_opset_version is None:
            versions = [version for version in map(self._opset_version, args) if version is not None]
            target_opset_version = max(versions) if versions else None

        op_list = []
        for op in args:
            op = super()._onnx_version_conversion(
                op, target_ir_version=target_ir_version, target_opset_version=target_opset_version
            )
            op_list.append(op)
        return op_list

    def combine(self, io_maps: Optional[list[list[tuple[str, str]]]] = None) -> onnx.ModelProto:
        """Combine the stored ONNX operators into one executable graph.

        Args:
            io_maps: Optional list with one list of ``(output, input)`` name pairs per joint between consecutive
                operators: ``io_maps[i]`` connects outputs of ``self.operators[i]`` to inputs of
                ``self.operators[i + 1]``, as in ``[[("out_a", "in_b")], [("out_b", "in_c")]]`` for three
                operators. Each entry may be any iterable of pairs; entries past the last joint are not read. When
                omitted, each joint connects the default ``"output"`` to ``"input"``.

        Returns:
            Combined ONNX model graph containing all operators in
            ``self.operators`` connected in sequence, with their metadata merged (the last value of each key).

        Raises:
            ValueCheckError: If ``io_maps`` has no entry for a joint, or an entry is not an iterable of name pairs.
                It is a :exc:`ValueError` too.

        Warns:
            UserWarning: If an output and the input it feeds have different tensor element types: the merged graph
                does not check a later model's declared input types.
        """
        return super()._combine(*self.operators, io_maps=io_maps)

    def create_session(
        self, providers: list[str] | None = None, session_options: Any | None = None
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime session for the combined graph.

        Args:
            providers: Optional ordered list of execution providers used by ONNX
                Runtime, for example GPU first and CPU as fallback.
            session_options: Optional ONNX Runtime session options controlling
                optimization, threading, and runtime behavior.

        Returns:
            ONNX Runtime inference session for ``self._combined_op``.
        """
        return super()._create_session(self._combined_op, providers, session_options)

    def export(self, file_path: str, **kwargs: Any) -> None:
        """Serialize the combined ONNX graph to an ``.onnx`` file.

        Args:
            file_path: Destination path where the composed ONNX graph is saved.
            kwargs: Additional keyword arguments forwarded to the export helper.
        """
        return super()._export(self._combined_op, file_path, **kwargs)

    def add_metadata(self, additional_metadata: Optional[list[tuple[str, str]]] = None) -> onnx.ModelProto:
        """Attach metadata entries to the combined ONNX graph.

        Args:
            additional_metadata: Optional list of ``(key, value)`` pairs to add
                to the ONNX model metadata properties.

        Returns:
            Combined ONNX model graph after metadata has been attached.
        """
        return super()._add_metadata(self._combined_op, additional_metadata)
