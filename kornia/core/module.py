from typing import Any, Callable, Optional, Union, List

import os
import math
import time
import datetime
from functools import wraps

from ._backend import Tensor, Module, from_numpy
from .external import numpy as np
from .external import PILImage as Image


class ImageModule(Module):
    """Handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end
    visualization, file saving features.
    """
    def __init__(self) -> None:
        super(ImageModule, self).__init__()
        self._output_image = None

    def convert_input_output(
        self, input_names_to_handle: Optional[List[Any]] = None, output_type: str = 'tensor'
    ) -> Callable:
        """Decorator to convert input and output types for a function.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # If input_names_to_handle is None, handle all inputs
                if input_names_to_handle is None:
                    # Convert all args to tensors
                    args = tuple(self.to_tensor(arg) if self._is_valid_arg(arg) else arg for arg in args)
                    # Convert all kwargs to tensors
                    kwargs = {k: self.to_tensor(v) if self._is_valid_arg(arg) else v for k, v in kwargs.items()}
                else:
                    # Convert specified args to tensors
                    args = list(args)
                    for i, (arg, name) in enumerate(zip(args, func.__code__.co_varnames)):
                        if name in input_names_to_handle:
                            args[i] = self.to_tensor(arg)
                    # Convert specified kwargs to tensors
                    for name, value in kwargs.items():
                        if name in input_names_to_handle:
                            kwargs[name] = self.to_tensor(value)

                # Call the actual forward method
                tensor_outputs = func(*args, **kwargs)

                if not isinstance(tensor_outputs, tuple):
                    tensor_outputs = (tensor_outputs,)

                # Convert outputs to the desired type
                outputs = []
                for output in tensor_outputs:
                    if output_type == 'tensor':
                        outputs.append(output)
                    elif output_type == 'numpy':
                        outputs.append(self.to_numpy(output))
                    elif output_type == 'pil':
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
        if isinstance(arg, (str, Tensor,)):
            return True
        # Make sure that the numpy and PIL are not necessarily needed to be imported.
        if isinstance(arg, (np.ndarray,)):
            return True
        if isinstance(arg, (Image.Image)):
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
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            return from_numpy(x).permute(2, 0, 1).float() / 255
        if isinstance(x, np.ndarray) and len(x.shape) == 4:
            return from_numpy(x).permute(0, 3, 1, 2).float() / 255
        elif isinstance(x, Image.Image):
            return from_numpy(np.array(x)).permute(2, 0, 1).float() / 255  # Convert PIL to tensor
        elif isinstance(x, str):
            return from_numpy(np.array(Image.open(x))).permute(2, 0, 1).float() / 255
        elif isinstance(x, Tensor):
            return x
        else:
            raise TypeError("Input type not supported")

    def to_numpy(self, x: Any) -> "np.ndarray":
        """Convert input to numpy array.

        Args:
            x: The input to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        if isinstance(x, Tensor):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, Image.Image):
            return np.array(x)
        else:
            raise TypeError("Input type not supported")

    def to_pil(self, x: Any) -> "Image.Image":
        """Convert input to PIL image.

        Args:
            x: The input to convert.

        Returns:
            Image.Image: The converted PIL image.
        """
        if isinstance(x, Tensor):
            x = x.cpu().detach().numpy() * 255
            if x.dim() == 3:
                x = x.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                return Image.fromarray(x.astype(np.uint8))
            elif x.dim() == 4:
                x = x.transpose(0, 2, 3, 1)  # Convert from (C, H, W) to (H, W, C)
                return [Image.fromarray(_x.astype(np.uint8)) for _x in x]
            else:
                raise NotImplementedError
        elif isinstance(x, np.ndarray):
            raise NotImplementedError
        elif isinstance(x, Image.Image):
            return x
        else:
            raise TypeError("Input type not supported")

    def show(self) -> Union["Image.Image", list["Image.Image"]]:
        """Returns PIL images.

        Returns:
            Image.Image or list[Image.Image]: The converted PIL images.
        """
        if self._output_image is None:
            self._output_image = self.__call__(*args, **kwargs)
        if len(self._output_image.shape) == 3:
            return Image.fromarray((self._output_image.permute(1, 2, 0).squeeze().numpy() * 255).astype(np.uint8))
        if len(self._output_image.shape) == 4:
            return [Image.fromarray(o) for o in (
                self._output_image.permute(0, 2, 3, 1).squeeze().numpy() * 255).astype(np.uint8)]
        raise ValueError

    def save(self, name: Optional[str] = None, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        """Saves the output image(s) to a directory.

        Args:
            name: Directory to save the images.
            rows: Number of rows for grid layout.
            cols: Number of columns for grid layout.
        """
        imgs = self.show()
        if name is None:
            name = f"Kornia-{str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S'))}.png"
        if isinstance(imgs, (list, tuple,)):
            if rows is None:
                rows = math.ceil(len(imgs) ** .5)
            if cols is None:
                cols = len(imgs) // rows
            w, h = imgs[0].size
            grid = Image.new('RGB', size=(cols * w, rows * h))
            grid_w, grid_h = grid.size
            for i, img in enumerate(imgs):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            grid.save(name)
        else:
            imgs.save(name)

    def __call__(
        self, *inputs, input_names_to_handle: Optional[List[Any]] = None, output_type: str = 'tensor', **kwargs
    ):
        """Overwrites the __call__ function to handle various inputs.

        Args:
            input_names_to_handle: List of input names to convert, if None, handle all inputs.
            output_type: Desired output type ('tensor', 'numpy', or 'pil').

        Returns:
            Callable: Decorated function with converted input and output types.
        """
        # Wrap the forward method with the decorator
        decorated_forward = self.convert_input_output(
            input_names_to_handle=input_names_to_handle, output_type=output_type
        )(super().__call__)
        self._output_image = decorated_forward(*inputs, **kwargs)
        return self._output_image
