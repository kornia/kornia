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

from typing import Any, Literal, Optional

from torch import nn

from .mixin.image_module import ImageModuleMixIn
from .mixin.onnx import ONNXExportMixin


class ImageModule(nn.Module, ImageModuleMixIn, ONNXExportMixin):
    """Handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end
    visualization, file saving features. Note that this module fits the classes that
    return one image tensor only.

    Note:
        The additional add-on features increase the use of memories. To restore the
        original behaviour, you may set `disable_features = True`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disable_features: bool = False

    @property
    def disable_features(self) -> bool:
        """Whether convenience I/O (input/output) helper features are disabled.

        This flag controls the extra behavior provided by :class:`ImageModuleMixIn`,
        such as automatic input conversion (for example, PIL or NumPy inputs),
        automatic output conversion, and output caching for visualization helpers.

        Returns:
            ``True`` if the helper features are bypassed and the module behaves like
            a plain :class:`torch.nn.Module` call. ``False`` if helper features are enabled.
        """
        return self._disable_features

    @disable_features.setter
    def disable_features(self, value: bool = True) -> None:
        """Enable or disable convenience I/O (input/output) handling features.

        Args:
            value: Feature toggle.
                - ``True``: disable automatic type conversion and output caching,
                  so ``__call__`` behaves closer to a raw PyTorch module call.
                - ``False``: keep helper features active.
        """
        self._disable_features = value

    def __call__(
        self,
        *inputs: Any,
        input_names_to_handle: Optional[list[Any]] = None,
        output_type: Literal["pt", "numpy", "pil"] = "pt",
        **kwargs: Any,
    ) -> Any:
        """Overwrite the __call__ function to handle various inputs.

        Args:
            inputs: Inputs to operate on.
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('pt', 'numpy', or 'pil').
            kwargs: Additional arguments.

        Returns:
            Callable: Decorated function with converted input and output types.

        """
        # Wrap the forward method with the decorator
        if not self._disable_features:
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super().__call__)
            _output_image = decorated_forward(*inputs, **kwargs)
            self._store_output_image(_output_image, output_type)
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image


class ImageSequential(nn.Sequential, ImageModuleMixIn, ONNXExportMixin):
    """Handles image-based operations as a sequential module.

    This modules accepts multiple input and output data types, provides end-to-end
    visualization, file saving features. Note that this module fits the classes that
    return one image tensor only.

    Note:
        The additional add-on features increase the use of memories. To restore the
        original behaviour, you may set `disable_features = True`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disable_features: bool = False

    @property
    def disable_features(self) -> bool:
        """Whether convenience I/O (input/output) helper features are disabled.

        This flag controls the extra behavior provided by :class:`ImageModuleMixIn`,
        such as automatic input conversion (for example, PIL or NumPy inputs),
        automatic output conversion, and output caching for visualization helpers.

        Returns:
            ``True`` if the helper features are bypassed and the module behaves like
            a plain :class:`torch.nn.Sequential` call. ``False`` if helper features are enabled.
        """
        return self._disable_features

    @disable_features.setter
    def disable_features(self, value: bool = True) -> None:
        """Enable or disable convenience I/O (input/output) handling features.

        Args:
            value: Feature toggle.
                - ``True``: disable automatic type conversion and output caching,
                  so ``__call__`` behaves closer to a raw PyTorch sequential call.
                - ``False``: keep helper features active.
        """
        self._disable_features = value

    def __call__(
        self,
        *inputs: Any,
        input_names_to_handle: Optional[list[Any]] = None,
        output_type: Literal["pt", "numpy", "pil"] = "pt",
        **kwargs: Any,
    ) -> Any:
        """Overwrite the __call__ function to handle various inputs.

        Args:
            inputs: Inputs to operate on.
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('pt', 'numpy', or 'pil').
            kwargs: Additional arguments.

        Returns:
            Callable: Decorated function with converted input and output types.

        """
        # Wrap the forward method with the decorator
        if not self._disable_features:
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super().__call__)
            _output_image = decorated_forward(*inputs, **kwargs)
            self._store_output_image(_output_image, output_type)
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image
