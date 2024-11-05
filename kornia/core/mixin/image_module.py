import datetime
import math
import os
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Union

from kornia.core._backend import Tensor, from_numpy
from kornia.core.external import PILImage as Image
from kornia.core.external import numpy as np


class ImageModuleMixIn:
    """A MixIn that handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end visualization, file saving features.
    Note that this MixIn fits the classes that return one image tensor only.
    """

    _output_image: Any

    def convert_input_output(
        self,
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: str = "tensor",
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
            from kornia.io import ImageLoadType, load_image  # pylint: disable=C0415

            return load_image(x, ImageLoadType.UNCHANGED) / 255
        if isinstance(x, (Tensor,)):
            return x
        if isinstance(x, (np.ndarray,)):  # type: ignore
            from kornia.utils.image import image_to_tensor  # pylint: disable=C0415

            return image_to_tensor(x) / 255
        if isinstance(x, (Image.Image,)):  # type: ignore
            return from_numpy(np.array(x)).permute(2, 0, 1).float() / 255  # type: ignore
        raise TypeError("Input type not supported")

    def to_numpy(self, x: Any) -> "np.array":  # type: ignore
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

    def to_pil(self, x: Any) -> "Image.Image":  # type: ignore
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
            from kornia.utils.image import make_grid  # pylint: disable=C0415

            if n_row is None:
                n_row = math.ceil(self._output_image.shape[0] ** 0.5)
            out_image = make_grid(self._output_image, n_row, padding=2)
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
        from kornia.io import write_image  # pylint: disable=C0415
        from kornia.utils.image import make_grid  # pylint: disable=C0415

        if name is None:
            name = f"Kornia-{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d%H%M%S')!s}.jpg"
        if len(self._output_image.shape) == 3:
            out_image = self._output_image
        if len(self._output_image.shape) == 4:
            if n_row is None:
                n_row = math.ceil(self._output_image.shape[0] ** 0.5)
            out_image = make_grid(self._output_image, n_row, padding=2)
        write_image(name, out_image.mul(255.0).byte())
