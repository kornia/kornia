import copy
import datetime
import math
import os
import io
from functools import wraps
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch

import kornia

from ._backend import Module, Sequential, Tensor, from_numpy, rand
from .external import PILImage as Image
from .external import numpy as np
from .external import onnx
from kornia.onnx.utils import add_metadata


class ONNXExportMixin:
    """Mixin class that provides ONNX export functionality for objects that support it.

    Attributes:
        ONNX_EXPORTABLE:
            A flag indicating whether the object can be exported to ONNX. Default is True.
        ONNX_DEFAULT_INPUTSHAPE:
            Default input shape for the ONNX export. A list of integers where `-1` indicates
            dynamic dimensions. Default is [-1, -1, -1, -1].
        ONNX_DEFAULT_OUTPUTSHAP:
            Default output shape for the ONNX export. A list of integers where `-1` indicates
            dynamic dimensions. Default is [-1, -1, -1, -1].
        ONNX_EXPORT_PSEUDO_SHAPE:
            This is used to create a dummy input tensor for the ONNX export. Default is [1, 3, 256, 256].
            It dimension shall match the ONNX_DEFAULT_INPUTSHAPE and ONNX_DEFAULT_OUTPUTSHAPE.
            Non-image dimensions are allowed.

    Note:
        - If `ONNX_EXPORTABLE` is False, indicating that the object cannot be exported to ONNX.
    """

    ONNX_EXPORTABLE: bool = True
    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = [-1, -1, -1, -1]
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = [-1, -1, -1, -1]
    ONNX_EXPORT_PSEUDO_SHAPE: ClassVar[List[int]] = [1, 3, 256, 256]
    ADDITIONAL_METADATA: ClassVar[List[Tuple[str, str]]] = []

    def to_onnx(
        self,
        onnx_name: Optional[str] = None,
        input_shape: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
        pseudo_shape: Optional[List[int]] = None,
        model: Optional[Module] = None,
        save: bool = True,
        additional_metadata: List[Tuple[str, str]] = [],
        **kwargs: Any
    ) -> "onnx.ModelProto":  # type: ignore
        """Exports the current object to an ONNX model file.

        Args:
            onnx_name:
                The name of the output ONNX file. If not provided, a default name in the
                format "Kornia-<ClassName>.onnx" will be used.
            input_shape:
                The input shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_INPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
            output_shape:
                The output shape for the model as a list of integers. If None,
                `ONNX_DEFAULT_OUTPUTSHAPE` will be used. Dynamic dimensions can be indicated by `-1`.
            pseudo_shape:
                The pseudo shape for the model as a list of integers. If None,
                `ONNX_EXPORT_PSEUDO_SHAPE` will be used.
            model:
                The model to export. If not provided, the current object will be used.
            save:
                If to save the model or load it.
            additional_metadata:
                Additional metadata to add to the ONNX model.
            **kwargs:
                Additional keyword arguments to pass to the `torch.onnx.export` function.

        Notes:
            - A dummy input tensor is created based on the provided or default input shape.
            - Dynamic axes for input and output tensors are configured where dimensions are marked `-1`.
            - The model is exported with `torch.onnx.export`, with constant folding enabled and opset version set to 17.
        """
        if not self.ONNX_EXPORTABLE:
            raise RuntimeError("This object cannot be exported to ONNX.")

        if input_shape is None:
            input_shape = self.ONNX_DEFAULT_INPUTSHAPE
        if output_shape is None:
            output_shape = self.ONNX_DEFAULT_OUTPUTSHAPE

        if onnx_name is None:
            onnx_name = f"Kornia-{self.__class__.__name__}.onnx"

        dummy_input = self._create_dummy_input(input_shape, pseudo_shape)
        dynamic_axes = self._create_dynamic_axes(input_shape, output_shape)

        default_args = dict(
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        default_args.update(kwargs)

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            model or self,  # type: ignore
            dummy_input,
            onnx_buffer,
            **default_args
        )
        onnx_buffer.seek(0)
        onnx_model = onnx.load(onnx_buffer)  # type: ignore

        additional_metadata = copy.deepcopy(additional_metadata)
        additional_metadata.extend(self.ADDITIONAL_METADATA)
        onnx_model = add_metadata(onnx_model, additional_metadata)
        if save:
            onnx.save(onnx_model, onnx_name)  # type: ignore
        return onnx_model

    def _create_dummy_input(
        self, input_shape: List[int], pseudo_shape: Optional[List[int]] = None
    ) -> Union[Tuple[Any, ...], Tensor]:
        return rand(*[(
            (self.ONNX_EXPORT_PSEUDO_SHAPE[i] if pseudo_shape is None else pseudo_shape[i]) if dim == -1 else dim
        ) for i, dim in enumerate(input_shape)])

    def _create_dynamic_axes(self, input_shape: List[int], output_shape: List[int]) -> Dict[str, Dict[int, str]]:
        return {
            "input": {i: "dim_" + str(i) for i, dim in enumerate(input_shape) if dim == -1},
            "output": {i: "dim_" + str(i) for i, dim in enumerate(output_shape) if dim == -1},
        }


class ImageModuleMixIn:
    """A MixIn that handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end visualization, file saving features.
    Note that this MixIn fits the classes that return one image tensor only.
    """

    _output_image: Any

    def convert_input_output(
        self, input_names_to_handle: Optional[List[Any]] = None, output_type: str = "tensor"
    ) -> Callable[[Any], Any]:
        """Decorator to convert input and output types for a function.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Union[Any, List[Any]]:
                # If input_names_to_handle is None, handle all inputs
                if input_names_to_handle is None:
                    # Convert all args to tensors
                    args = tuple(self.to_tensor(arg) if self._is_valid_arg(arg) else arg for arg in args)
                    # Convert all kwargs to tensors
                    kwargs = {k: self.to_tensor(v) if self._is_valid_arg(v) else v for k, v in kwargs.items()}
                else:
                    # Convert specified args to tensors
                    args = list(args)  # type:ignore
                    for i, (arg, name) in enumerate(zip(args, func.__code__.co_varnames)):
                        if name in input_names_to_handle:
                            args[i] = self.to_tensor(arg)  # type:ignore
                    # Convert specified kwargs to tensors
                    for name, value in kwargs.items():
                        if name in input_names_to_handle:
                            kwargs[name] = self.to_tensor(value)

                # Call the actual forward method
                tensor_outputs = func(*args, **kwargs)

                if not isinstance(tensor_outputs, (tuple,)):
                    tensor_outputs = (tensor_outputs,)

                # Convert outputs to the desired type
                outputs = []
                for output in tensor_outputs:
                    if output_type == "tensor":
                        outputs.append(output)
                    elif output_type == "numpy":
                        outputs.append(self.to_numpy(output))
                    elif output_type == "pil":
                        outputs.append(self.to_pil(output))
                    else:
                        raise ValueError("Output type not supported. Choose from 'tensor', 'numpy', or 'pil'.")

                return outputs if len(outputs) > 1 else outputs[0]

            return wrapper

        return decorator

    def _is_valid_arg(self, arg: Any) -> bool:
        """Check if the argument is a valid type for conversion.

        Args:
            arg: The argument to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        if isinstance(arg, (str,)) and os.path.exists(arg):
            return True
        if isinstance(arg, (Tensor,)):
            return True
        # Make sure that the numpy and PIL are not necessarily needed to be imported.
        if isinstance(arg, (np.ndarray,)):  # type: ignore
            return True
        if isinstance(arg, (Image.Image)):  # type: ignore
            return True
        return False

    def to_tensor(self, x: Any) -> Tensor:
        """Convert input to tensor.

        Supports image path, numpy array, PIL image, and raw tensor.

        Args:
            x: The input to convert.

        Returns:
            Tensor: The converted tensor.
        """
        if isinstance(x, (str,)):
            return kornia.io.load_image(x, kornia.io.ImageLoadType.UNCHANGED) / 255
        if isinstance(x, (Tensor,)):
            return x
        if isinstance(x, (np.ndarray,)):  # type: ignore
            return kornia.utils.image.image_to_tensor(x) / 255
        if isinstance(x, (Image.Image,)):  # type: ignore
            return from_numpy(np.array(x)).permute(2, 0, 1).float() / 255  # type: ignore
        raise TypeError("Input type not supported")

    def to_numpy(self, x: Any) -> np.array:  # type: ignore
        """Convert input to numpy array.

        Args:
            x: The input to convert.

        Returns:
            np.array: The converted numpy array.
        """
        if isinstance(x, (Tensor,)):
            return x.cpu().detach().numpy()
        if isinstance(x, (np.ndarray,)):  # type: ignore
            return x
        if isinstance(x, (Image.Image,)):  # type: ignore
            return np.array(x)  # type: ignore
        raise TypeError("Input type not supported")

    def to_pil(self, x: Any) -> Image.Image:  # type: ignore
        """Convert input to PIL image.

        Args:
            x: The input to convert.

        Returns:
            Image.Image: The converted PIL image.
        """
        if isinstance(x, (Tensor,)):
            x = x.cpu().detach() * 255
            if x.dim() == 3:
                x = x.permute(1, 2, 0)
                return Image.fromarray(x.byte().numpy())  # type: ignore
            elif x.dim() == 4:
                x = x.permute(0, 2, 3, 1)
                return [Image.fromarray(_x.byte().numpy()) for _x in x]  # type: ignore
            else:
                raise NotImplementedError
        if isinstance(x, (np.ndarray,)):  # type: ignore
            raise NotImplementedError
        if isinstance(x, (Image.Image,)):  # type: ignore
            return x
        raise TypeError("Input type not supported")

    def _detach_tensor_to_cpu(
        self, output_image: Union[Tensor, List[Tensor], Tuple[Tensor]]
    ) -> Union[Tensor, List[Tensor], Tuple[Tensor]]:
        if isinstance(output_image, (Tensor,)):
            return output_image.detach().cpu()
        if isinstance(
            output_image,
            (
                list,
                tuple,
            ),
        ):
            return type(output_image)([self._detach_tensor_to_cpu(out) for out in output_image])  # type: ignore
        raise RuntimeError(f"Unexpected object {output_image} with a type of `{type(output_image)}`")

    def show(self, n_row: Optional[int] = None, backend: str = "pil", display: bool = True) -> Optional[Any]:
        """Returns PIL images.

        Args:
            n_row: Number of images displayed in each row of the grid.
            backend: visualization backend. Only PIL is supported now.
        """
        if self._output_image is None:
            raise ValueError("No pre-computed images found. Needs to execute first.")

        if len(self._output_image.shape) == 3:
            out_image = self._output_image
        elif len(self._output_image.shape) == 4:
            if n_row is None:
                n_row = math.ceil(self._output_image.shape[0] ** 0.5)
            out_image = kornia.utils.image.make_grid(self._output_image, n_row, padding=2)
        else:
            raise ValueError

        if backend == "pil" and display:
            Image.fromarray((out_image.permute(1, 2, 0).squeeze().numpy() * 255).astype(np.uint8)).show()  # type: ignore
            return None
        if backend == "pil":
            return Image.fromarray((out_image.permute(1, 2, 0).squeeze().numpy() * 255).astype(np.uint8))  # type: ignore
        raise ValueError(f"Unsupported backend `{backend}`.")

    def save(self, name: Optional[str] = None, n_row: Optional[int] = None) -> None:
        """Saves the output image(s) to a directory.

        Args:
            name: Directory to save the images.
            n_row: Number of images displayed in each row of the grid.
        """
        if name is None:
            name = f"Kornia-{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d%H%M%S')!s}.jpg"
        if len(self._output_image.shape) == 3:
            out_image = self._output_image
        if len(self._output_image.shape) == 4:
            if n_row is None:
                n_row = math.ceil(self._output_image.shape[0] ** 0.5)
            out_image = kornia.utils.image.make_grid(self._output_image, n_row, padding=2)
        kornia.io.write_image(name, out_image.mul(255.0).byte())


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
