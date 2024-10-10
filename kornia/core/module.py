from typing import Any, List, Optional

from ._backend import Module, Sequential
from .mixin import ImageModuleMixIn, ONNXExportMixin


class ImageModule(Module, ImageModuleMixIn, ONNXExportMixin):
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
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: str = "tensor",
        **kwargs: Any,
    ) -> Any:
        """Overwrites the __call__ function to handle various inputs.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """

        # Wrap the forward method with the decorator
        if not self._disable_features:
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super().__call__)
            _output_image = decorated_forward(*inputs, **kwargs)
            if output_type == "tensor":
                self._output_image = self._detach_tensor_to_cpu(_output_image)
            else:
                self._output_image = _output_image
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image


class ImageSequential(Sequential, ImageModuleMixIn, ONNXExportMixin):
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
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: str = "tensor",
        **kwargs: Any,
    ) -> Any:
        """Overwrites the __call__ function to handle various inputs.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """

        # Wrap the forward method with the decorator
        if not self._disable_features:
            decorated_forward = self.convert_input_output(
                input_names_to_handle=input_names_to_handle, output_type=output_type
            )(super().__call__)
            _output_image = decorated_forward(*inputs, **kwargs)
            if output_type == "tensor":
                self._output_image = self._detach_tensor_to_cpu(_output_image)
            else:
                self._output_image = _output_image
        else:
            _output_image = super().__call__(*inputs, **kwargs)
        return _output_image
