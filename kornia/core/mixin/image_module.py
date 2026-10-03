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

import datetime
import math
import os
import sys
from functools import wraps
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import torch

from kornia.core.external import PILImage as Image
from kornia.core.external import numpy as np
from kornia.core.utils import is_exporting


def _image_to_float(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor to floating point, scaling an integer image by the maximum of its dtype.

    A ``uint8`` image is divided by 255, a ``uint16`` one by 65535, and so on; a ``bool`` image becomes 0 and 1; a
    floating image is returned unchanged, since it is taken to be in ``[0, 1]`` already. Integer and ``bool`` images
    become the default floating dtype. A signed integer image maps to ``[iinfo.min / iinfo.max, 1]``: negative values
    stay negative, and the minimum lands just below -1 (``int8`` -128 gives -128 / 127). The division runs in float32
    when the default dtype is narrower, so a ``float16`` default still maps the ``uint16`` maximum 65535, which is
    above the largest finite ``float16`` (65504), to 1 rather than ``inf``.
    """
    if image.is_floating_point():
        return image
    default_dtype = torch.get_default_dtype()
    if image.dtype == torch.bool:
        return image.to(default_dtype)
    working_dtype = torch.promote_types(default_dtype, torch.float32)
    return (image.to(working_dtype) / float(torch.iinfo(image.dtype).max)).to(default_dtype)


def _array_to_float_image(array: Any) -> torch.Tensor:
    """Convert a channels-last NumPy image to a channels-first tensor scaled by :func:`_image_to_float`."""
    from kornia.image.utils import image_to_tensor  # pylint: disable=C0415

    if not array.dtype.isnative:  # torch.from_numpy rejects big-endian data, such as PIL's "I;16B" mode
        array = array.astype(array.dtype.newbyteorder("="))
    return _image_to_float(image_to_tensor(array))


# Non-``uint8`` dtypes PIL stores as a one-channel image (modes "1", "I;16" and "I"); ``uint8`` takes 1 to 4 channels.
_PIL_ONE_CHANNEL_DTYPES = (torch.bool, torch.int8, torch.int16, torch.uint16, torch.int32, torch.uint32)


def _to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a floating image in ``[0, 1]`` to ``uint8`` for display or an 8-bit file.

    Values are clamped to ``[0, 1]``, scaled by 255 and rounded, so an image that overshoots the range saturates
    instead of wrapping modulo 256 (1.1 would otherwise become 24, black where it should be white), and a negative
    value becomes 0 on every torch version. A half-precision image is scaled in float32, so the product is not rounded
    to the half dtype before ``round`` (``float16`` 0.0058823 is 1.49998 / 255 and gives 1, not 2). Non-floating
    images pass through untouched, so a ``uint8`` or ``uint16`` image keeps its values.
    """
    if not image.is_floating_point():
        return image
    working_dtype = torch.promote_types(image.dtype, torch.float32)
    return (image.detach().to(working_dtype).clamp(0.0, 1.0) * 255).round().to(torch.uint8)


class ImageModuleMixIn:
    """A MixIn that handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end visualization, file saving features.
    Note that this MixIn fits the classes that return one image tensor only.

    Non-tensor inputs are converted by :meth:`to_tensor`: a NumPy array of shape :math:`(H, W)`, :math:`(H, W, C)`
    or :math:`(B, H, W, C)`, a PIL image or an image path becomes a :math:`(C, H, W)` or :math:`(B, C, H, W)` tensor.
    An integer image is scaled by the maximum of its dtype (``uint8`` by 255, ``uint16`` by 65535; a signed image keeps
    its negative values), a ``bool`` image becomes 0 and 1, and a floating image keeps its values. Tensors pass through
    unchanged. ``output_type="numpy"`` returns channels-last arrays with the values of the output tensor, so a floating
    output fed back in converts to the same tensor. ``output_type="pil"``, :meth:`show` and :meth:`save` clamp a
    floating image to ``[0, 1]`` and round it to 8 bits on the CPU, and pass a ``uint8`` image through with its values.
    """

    _output_image: Any

    def convert_input_output(
        self,
        input_names_to_handle: Optional[List[Any]] = None,
        output_type: Literal["pt", "numpy", "pil"] = "pt",
        *,
        cache_output: bool = False,
    ) -> Callable[[Any], Any]:
        """Convert input and output types for a function.

        Args:
            input_names_to_handle: List of input names to convert.
                If None, convert every tensor, NumPy array and PIL image argument, and load a string as an image
                path only if it is the first positional argument.
            output_type: Desired output type ('pt', 'numpy', or 'pil').
            cache_output: Cache detached tensor outputs before converting their type, for visualization helpers.

        Returns:
            Callable: Decorated function with converted input and output types.

        """
        # Validate output_type at the start
        self._check_output_type(output_type)

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Union[Any, List[Any]]:
                tensor_outputs = self._call_converted(func, args, kwargs, input_names_to_handle, "pt")
                if cache_output:
                    self._store_output_image(self._convert_output(tensor_outputs, "pt"), "pt")
                return self._convert_output(tensor_outputs, output_type)

            return wrapper

        return decorator

    def _call_converted(
        self,
        func: Callable[[Any], Any],
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
        input_names_to_handle: Optional[List[Any]],
        output_type: Literal["pt", "numpy", "pil"],
    ) -> Union[Any, List[Any]]:
        if input_names_to_handle is None:
            args = tuple(
                self.to_tensor(arg) if (i == 0 or not isinstance(arg, str)) and self._is_valid_arg(arg) else arg
                for i, arg in enumerate(args)
            )
            kwargs = {
                k: self.to_tensor(v) if not isinstance(v, str) and self._is_valid_arg(v) else v
                for k, v in kwargs.items()
            }
        else:
            args = list(args)
            for i, (arg, name) in enumerate(zip(args, func.__code__.co_varnames)):  # ty: ignore[unresolved-attribute]
                if name in input_names_to_handle:
                    args[i] = self.to_tensor(arg)  # type:ignore
            for name, value in kwargs.items():
                if name in input_names_to_handle:
                    kwargs[name] = self.to_tensor(value)

        return func(*args, **kwargs)

    @staticmethod
    def _check_output_type(output_type: str) -> None:
        if output_type not in ("pt", "numpy", "pil"):
            raise ValueError(f"Invalid output_type '{output_type}'. Must be one of 'pt', 'numpy', or 'pil'.")

    def _convert_output(self, tensor_outputs: Any, output_type: str) -> Any:
        """Convert a forward output to ``output_type`` the way :meth:`convert_input_output` does.

        Args:
            tensor_outputs: The forward output: a tensor, or a tuple whose elements are converted one by one.
            output_type: Desired output type ('pt', 'numpy', or 'pil').

        Returns:
            The converted output, or a list of converted outputs for a tuple of several.

        """
        if not isinstance(tensor_outputs, tuple):
            tensor_outputs = (tensor_outputs,)

        outputs = []
        for output in tensor_outputs:
            if output_type == "pt":
                outputs.append(output)
            elif output_type == "numpy":
                outputs.append(self.to_numpy(output))
            elif output_type == "pil":
                outputs.append(self.to_pil(output))
            else:
                raise ValueError("Output type not supported. Choose from 'pt', 'numpy', or 'pil'.")

        return outputs if len(outputs) > 1 else outputs[0]

    def _is_valid_arg(self, arg: Any) -> bool:
        """Check if the argument is a valid type for conversion.

        Args:
            arg: The argument to check.

        Returns:
            bool: True if valid, False otherwise.

        """
        if isinstance(arg, str) and os.path.exists(arg):
            return True
        if isinstance(arg, torch.Tensor):
            return True
        if isinstance(arg, np.ndarray):  # type: ignore
            return True
        # A PIL image exists only once PIL is imported: look the module up instead of importing it through the lazy
        # loader, which would raise on an install without the "image" extra for every non-image argument.
        pil_image = sys.modules.get("PIL.Image")
        if pil_image is not None and isinstance(arg, pil_image.Image):
            return True
        return False

    def to_tensor(self, x: Any) -> torch.Tensor:
        """Convert input to tensor.

        Supports image path, numpy array, PIL image, and raw tensor. A NumPy array of shape :math:`(H, W)`,
        :math:`(H, W, C)` or :math:`(B, H, W, C)` becomes :math:`(C, H, W)` or :math:`(B, C, H, W)`, with one channel
        for :math:`(H, W)`; a PIL image converts like its NumPy array, except that a palette image (mode ``P`` or
        ``PA``) is converted to RGB, or RGBA when it has transparency, first. An integer image is divided by the maximum
        of its dtype (``uint8`` by 255, ``uint16`` by 65535, a PIL mode ``I`` image by the ``int32`` maximum); a signed
        image maps to ``[iinfo.min / iinfo.max, 1]``, so its negative values stay negative. A ``bool`` image becomes 0
        and 1, and a floating image keeps its values; integer and ``bool`` images become the default floating dtype. A
        tensor is returned unchanged.

        Args:
            x: The input to convert.

        Returns:
            Tensor: The converted tensor.

        """
        if isinstance(x, str):
            from kornia.io import ImageLoadType, load_image  # pylint: disable=C0415

            return _image_to_float(load_image(x, ImageLoadType.UNCHANGED))
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):  # type: ignore
            return _array_to_float_image(x)
        if isinstance(x, Image.Image):  # type: ignore
            if x.mode in ("P", "PA"):  # palette indices are not intensities
                x = x.convert("RGBA" if x.mode == "PA" or "transparency" in x.info else "RGB")
            return _array_to_float_image(np.array(x))  # type: ignore
        raise TypeError("Input type not supported")

    def to_numpy(self, x: Any) -> "np.array":  # type: ignore
        """Convert input to numpy array.

        A :math:`(C, H, W)` or :math:`(B, C, H, W)` tensor becomes a channels-last :math:`(H, W, C)` or
        :math:`(B, H, W, C)` array with the same values, the layout :meth:`to_tensor` accepts, so a floating array
        converts back to the same tensor. A tensor of any other rank keeps its shape. Every 3-D or 4-D tensor is taken
        to be an image, so a non-image element of a tuple output, such as a :math:`(B, N, 4)` box tensor, is moved to
        channels-last too; modules with several outputs are tracked in
        `#5210 <https://github.com/kornia/kornia/issues/5210>`_.

        Args:
            x: The input to convert.

        Returns:
            np.array: The converted numpy array.

        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.dim() == 3:
                x = x.permute(1, 2, 0)
            elif x.dim() == 4:
                x = x.permute(0, 2, 3, 1)
            return x.contiguous().numpy()
        if isinstance(x, np.ndarray):  # type: ignore
            return x
        if isinstance(x, Image.Image):  # type: ignore
            return np.array(x)  # type: ignore
        raise TypeError("Input type not supported")

    def to_pil(self, x: Any) -> "Image.Image":  # type: ignore
        """Convert input to PIL image.

        A floating image is clamped to ``[0, 1]`` and rounded to 8 bits on the CPU; a ``uint8`` image keeps its
        values. A one-channel image becomes a mode ``"L"`` image, and a :math:`(B, C, H, W)` batch a list of images.
        Other integer and ``bool`` images convert only with one channel (PIL modes ``"I"``, ``"I;16"`` and ``"1"``).

        Args:
            x: The input to convert.

        Returns:
            Image.Image: The converted PIL image.

        """
        if isinstance(x, torch.Tensor):
            if x.dim() == 3:
                return self._chw_to_pil(x)
            if x.dim() == 4:
                return [self._chw_to_pil(_x) for _x in x]  # type: ignore
            raise NotImplementedError(
                f"to_pil converts a (C, H, W) or (B, C, H, W) tensor, got a tensor of shape {tuple(x.shape)}."
            )
        if isinstance(x, np.ndarray):  # type: ignore
            raise NotImplementedError("to_pil does not convert NumPy arrays; convert them with to_tensor first.")
        if isinstance(x, Image.Image):  # type: ignore
            return x
        raise TypeError("Input type not supported")

    @staticmethod
    def _chw_to_pil(image: torch.Tensor) -> "Image.Image":  # type: ignore
        image = _to_uint8_image(image.detach().cpu())
        if image.dtype != torch.uint8 and (image.shape[0] != 1 or image.dtype not in _PIL_ONE_CHANNEL_DTYPES):
            raise NotImplementedError(
                "to_pil converts a float or uint8 image, or a one-channel bool, int8, int16, int32, uint16 or uint32 "
                f"image; got a {image.shape[0]}-channel {image.dtype} tensor."
            )
        if image.shape[0] == 1:
            return Image.fromarray(image[0].numpy())  # type: ignore
        return Image.fromarray(image.permute(1, 2, 0).numpy())  # type: ignore

    def _detach_tensor(
        self, output_image: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]:
        if isinstance(output_image, torch.Tensor):
            return output_image.detach()
        if isinstance(output_image, (list, tuple)):
            return type(output_image)([self._detach_tensor(out) for out in output_image])  # type: ignore
        raise RuntimeError(f"Unexpected object {output_image} with a type of `{type(output_image)}`")

    def _store_output_image(self, output_image: Any, output_type: str) -> None:
        """Cache detached outputs on their device; ``.show()`` / ``.save()`` move them to CPU on use.

        Skipped inside a ``torch.export`` capture: caching mutates module state in ``forward``,
        which ``torch.export`` (torch <= 2.9) rejects. The captured output is unaffected.
        """
        if is_exporting():
            return
        self._output_image = self._detach_tensor(output_image) if output_type == "pt" else output_image

    def show(self, n_row: Optional[int] = None, backend: str = "pil", display: bool = True) -> Optional[Any]:
        """Return PIL images.

        Args:
            n_row: Number of images displayed in each row of the grid.
            backend: visualization backend. Only PIL is supported now.
            display: Whether or not to show the image.

        """
        if self._output_image is None:
            raise ValueError("No pre-computed images found. Needs to execute first.")
        output_image = self._output_image
        if isinstance(output_image, torch.Tensor):
            output_image = output_image.detach().cpu()

        if len(output_image.shape) == 3:
            out_image = output_image
        elif len(output_image.shape) == 4:
            from kornia.image.utils import make_grid  # pylint: disable=C0415

            if n_row is None:
                n_row = math.ceil(output_image.shape[0] ** 0.5)
            out_image = make_grid(output_image, n_row, padding=2)
        else:
            raise ValueError

        if backend == "pil" and display:
            Image.fromarray(_to_uint8_image(out_image).permute(1, 2, 0).squeeze().numpy()).show()  # type: ignore
            return None
        if backend == "pil":
            return Image.fromarray(_to_uint8_image(out_image).permute(1, 2, 0).squeeze().numpy())  # type: ignore
        raise ValueError(f"Unsupported backend `{backend}`.")

    def save(self, name: Optional[str] = None, n_row: Optional[int] = None) -> None:
        """Save the output image(s) to a directory.

        Args:
            name: Directory to save the images.
            n_row: Number of images displayed in each row of the grid.

        """
        from kornia.image.utils import make_grid  # pylint: disable=C0415
        from kornia.io import write_image  # pylint: disable=C0415

        if self._output_image is None:
            raise ValueError("No pre-computed images found. Needs to execute first.")
        output_image = self._output_image
        if isinstance(output_image, torch.Tensor):
            output_image = output_image.detach().cpu()

        if name is None:
            name = f"Kornia-{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d%H%M%S')!s}.jpg"
        if len(output_image.shape) == 3:
            out_image = output_image
        if len(output_image.shape) == 4:
            if n_row is None:
                n_row = math.ceil(output_image.shape[0] ** 0.5)
            out_image = make_grid(output_image, n_row, padding=2)
        write_image(name, _to_uint8_image(out_image))
