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

# ImageModuleMixIn is now imported from .mixin.image_module
# The class definition was moved there to avoid duplication


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
        return self._disable_features

    @disable_features.setter
    def disable_features(self, value: bool = True) -> None:
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
            if output_type == "pt":
                self._output_image = self._detach_tensor_to_cpu(_output_image)
            else:
                self._output_image = _output_image
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
        return self._disable_features

    @disable_features.setter
    def disable_features(self, value: bool = True) -> None:
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
            if output_type == "pt":
                self._output_image = self._detach_tensor_to_cpu(_output_image)
            else:
                self._output_image = _output_image
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image
