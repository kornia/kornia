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

"""Chain several ONNX models into one merged graph and run it with one ONNX Runtime session.

Which class to use:

- Models run one after another, the outputs of each feeding the inputs of the next: :class:`ONNXSequential`.
- One model: :class:`~kornia.onnx.module.ONNXModule`.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from kornia.core.external import onnx
from kornia.core.external import onnxruntime as ort
from kornia.core.mixin import ONNXMixin, ONNXRuntimeMixin

__all__ = ["ONNXSequential"]


class ONNXSequential(ONNXMixin, ONNXRuntimeMixin):
    """Chain several ONNX models into one graph and run it with one ONNX Runtime session.

    ``seq = ONNXSequential(model_a, model_b, ...)`` merges the models into one graph, each joint connecting the output
    named ``"output"`` to the input named ``"input"`` unless ``io_maps`` says otherwise, and opens one session on it;
    ``outputs = seq(*inputs)`` takes inputs as :class:`~kornia.onnx.module.ONNXModule` does, one per unconnected
    input, and returns a list of numpy arrays. The names of model ``i`` get the prefix ``K<i>-`` (``K00-input``,
    ``K01-output``), and an output that feeds nothing stays a graph output.

    Args:
        *args: The ONNX models, in the order they run, each as :class:`~kornia.onnx.module.ONNXModule` takes it: an
            ONNX ModelProto object, a file path as ``str`` or :class:`os.PathLike` (a path object always names a
            local file), or a ``str`` HTTP(S) URL or ``hf://<folder>/<name>`` name.
        providers: A list of execution providers for ONNXRuntime
            (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']). A provider that the installed onnxruntime
            build does not offer is skipped with onnxruntime's ``UserWarning``; the session runs on the CPU if none
            is left. If None, the CPU.
        session_options: ONNX Runtime ``SessionOptions``, used as given. If None, the session uses
            ``ORT_ENABLE_EXTENDED`` graph optimizations.
        io_maps: An optional list with one list of ``(output, input)`` name pairs per joint between consecutive
            models: ``io_maps[0]`` connects outputs of the first model to inputs of the second, ``io_maps[1]`` the
            second to the third, and so on, as in ``[[("out_a", "in_b")], [("out_b", "in_c")]]`` for three models.
            Each entry may be any iterable of pairs (a ``zip``, ``dict.items()``, ...), and ``io_maps`` anything
            indexed by the joint; entries past the last joint are not read.
        cache_dir: The directory where downloaded ONNX models (``hf://`` names and URLs) are cached, as
            :meth:`~kornia.onnx.utils.ONNXLoader.load_model` lays it out. If None,
            ``kornia.config.kornia_config.hub_onnx_dir`` (``.kornia_hub/onnx_models`` under the working directory).
        auto_ir_version_conversion: If True, convert the models to one IR version and one OPSET version before
            combining them: IR version 9, and the highest OPSET version among the models, to which the others are
            converted up.
        target_ir_version: The IR version to convert every model to before combining them, whether or not
            `auto_ir_version_conversion` is set. If None, the models keep their IR versions, unless
            `auto_ir_version_conversion` converts them to 9.
        target_opset_version: The OPSET version to convert every model to before combining them, whether or not
            `auto_ir_version_conversion` is set. If None, the models keep their OPSET versions, unless
            `auto_ir_version_conversion` converts them to the highest among them. A conversion that fails raises the
            error of the ONNX version converter, of ``onnx.checker`` or of ONNX Runtime.
        force_download: If True, every ``hf://`` name and URL is downloaded again, and replaces the cached file, even
            when it is cached. By default a cached file is used as it is, however old, and one that does not load
            fails every time until it is deleted. It has no effect on a model object or a path to a local file.

    Raises:
        TypeError: If a model is not a ModelProto, a ``str`` or an :class:`os.PathLike`.
        ValueError: If no model is given, if a local file does not exist, if a joint, set by ``io_maps`` or by
            default, names an output or input a model does not have, if ``io_maps`` has no entry for a joint or an
            ``io_maps`` entry is not name pairs, or if the models differ in IR version or opset and no conversion makes
            them agree.

    Warns:
        UserWarning: If an output and the input it feeds have different tensor element types; declared dimensions are
            not compared.

    See Also:
        - :class:`~kornia.onnx.module.ONNXModule`: one model, with its own input and output names.
        - :func:`~kornia.onnx.utils.io_name_conversion`: rename a model's inputs and outputs to ``"input"`` and
          ``"output"``, so that it chains without ``io_maps``.

    Example:
        >>> import numpy as np
        >>> from onnx import TensorProto, helper
        >>> from kornia.onnx import ONNXSequential
        >>> def unary(op):
        ...     graph = helper.make_graph(
        ...         [helper.make_node(op, ["input"], ["output"])], op,
        ...         [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])],
        ...         [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3])])
        ...     return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        >>> seq = ONNXSequential(unary("Neg"), unary("Relu"))
        >>> [i.name for i in seq.get_session().get_inputs()], [o.name for o in seq.get_session().get_outputs()]
        (['K00-input'], ['K01-output'])
        >>> seq(np.array([[-1.0, 2.0, -3.0]], dtype=np.float32))
        [array([[1., 0., 3.]], dtype=float32)]

    """

    def __init__(
        self,
        *args: Union[onnx.ModelProto, str, os.PathLike],
        providers: Optional[list[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
        io_maps: Optional[list[list[tuple[str, str]]]] = None,
        cache_dir: Optional[str] = None,
        auto_ir_version_conversion: bool = False,
        target_ir_version: Optional[int] = None,
        target_opset_version: Optional[int] = None,
        force_download: bool = False,
    ) -> None:
        self.operators = self._load_ops(*args, cache_dir=cache_dir, force_download=force_download)
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
        """Merge the stored ONNX models into one graph, without changing the object's session.

        The constructor calls it once; a call after construction returns a new graph and leaves the running one as
        it was.

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
            ValueError: From ``onnx.compose.merge_models``, if a joint names an output or input that an operator does
                not have, or the operators differ in IR version or opset.

        Warns:
            UserWarning: If an output and the input it feeds have different tensor element types: the merged graph
                does not check a later model's declared input types.
        """
        return super()._combine(*self.operators, io_maps=io_maps)

    def create_session(
        self, providers: list[str] | None = None, session_options: Any | None = None
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime session for the combined graph, without making it the object's session.

        Pass the result to ``set_session`` to run the object on it.

        Args:
            providers: Ordered list of ONNX Runtime execution providers, such as ``"CUDAExecutionProvider"`` followed
                by ``"CPUExecutionProvider"``. A provider the installed onnxruntime build does not offer is skipped
                with onnxruntime's ``UserWarning``. If None, the CPU.
            session_options: ONNX Runtime ``SessionOptions``, used as given. If None, ``ORT_ENABLE_EXTENDED`` graph
                optimizations.

        Returns:
            ONNX Runtime inference session for the combined graph.
        """
        return super()._create_session(self._combined_op, providers, session_options)

    def export(self, file_path: str, **kwargs: Any) -> None:
        """Save the combined ONNX graph to an ``.onnx`` file with ``onnx.save``.

        Args:
            file_path: Destination path for the ``.onnx`` file.
            kwargs: Keyword arguments of ``onnx.save``, such as ``save_as_external_data``.
        """
        return super()._export(self._combined_op, file_path, **kwargs)

    def add_metadata(self, additional_metadata: Optional[list[tuple[str, str]]] = None) -> onnx.ModelProto:
        """Attach metadata entries to the combined ONNX graph.

        The metadata is written into the combined graph in place, as :func:`~kornia.onnx.utils.add_metadata` writes
        it. The running session keeps the metadata it was created with; :meth:`export` saves the new entries.

        Args:
            additional_metadata: Optional list of ``(key, value)`` pairs to add
                to the ONNX model metadata properties.

        Returns:
            Combined ONNX model graph after metadata has been attached.
        """
        return super()._add_metadata(self._combined_op, additional_metadata)
