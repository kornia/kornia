"""The testing package contains testing-specific utilities."""
from typing import List, Optional, Sequence, cast

from typing_extensions import TypeGuard

from kornia.core import Tensor

__all__ = [
    "KORNIA_CHECK_SHAPE",
    "KORNIA_CHECK",
    "KORNIA_UNWRAP",
    "KORNIA_CHECK_TYPE",
    "KORNIA_CHECK_IS_TENSOR",
    "KORNIA_CHECK_IS_LIST_OF_TENSOR",
    "KORNIA_CHECK_SAME_DEVICE",
    "KORNIA_CHECK_SAME_DEVICES",
    "KORNIA_CHECK_IS_COLOR",
    "KORNIA_CHECK_IS_GRAY",
    "KORNIA_CHECK_DM_DESC",
    "KORNIA_CHECK_LAF",
]

# Logger api

# TODO: add somehow type check, or enforce to do it before
# TODO: get rid of torchscript test because prevents us to have type safe code


def KORNIA_CHECK_SHAPE(x: Tensor, shape: List[str]) -> None:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.

    Raises:
        Exception: if the input tensor is has not the expected shape.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["B","C", "H", "W"])  # implicit

        >>> x = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_SHAPE(x, ["2","3", "H", "W"])  # explicit
    """

    if '*' == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif '*' == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")


def KORNIA_CHECK(condition: bool, msg: Optional[str] = None) -> None:
    """Check any arbitrary boolean condition.

    Args:
        condition: the condition to evaluate.
        msg: message to show in the exception.

    Raises:
        Exception: if the confition is met.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK(x.shape[-2:] == (3, 3), "Invalid homography")
    """
    if not condition:
        raise Exception(f"{condition} not true.\n{msg}")


def KORNIA_UNWRAP(maybe_obj, typ):
    """Unwraps an optional contained value that may or not be present.

    Args:
        maybe_obj: the object to unwrap.
        typ: expected type after unwrap.
    """
    return cast(typ, maybe_obj)  # type: ignore # TODO: this function will change after kornia/pr#1987


def KORNIA_CHECK_TYPE(x, typ, msg: Optional[str] = None):
    """Check the type of an aribratry variable.

    Args:
        x: any input variable.
        typ: the expected type of the variable.
        msg: message to show in the exception.

    Raises:
        TypeException: if the input variable does not match with the expected.

    Example:
        >>> KORNIA_CHECK_TYPE("foo", str, "Invalid string")
    """
    if not isinstance(x, typ):
        raise TypeError(f"Invalid type: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_TENSOR(x, msg: Optional[str] = None):
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.

    Raises:
        TypeException: if the input variable does not match with the expected.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_TENSOR(x, "Invalid tensor")
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_LIST_OF_TENSOR(x: Optional[Sequence[object]]) -> TypeGuard[List[Tensor]]:
    """Check the input variable is a List of Tensors.

    Args:
        x: Any sequence of objects

    Return:
        True if the input is a list of Tensors, otherwise return False.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR(x)
        False
        >>> KORNIA_CHECK_IS_LIST_OF_TENSOR([x])
        True
    """
    return isinstance(x, list) and all(isinstance(d, Tensor) for d in x)


def KORNIA_CHECK_SAME_DEVICE(x: Tensor, y: Tensor):
    """Check whether two tensor in the same device.

    Args:
        x: first tensor to evaluate.
        y: sencod tensor to evaluate.
        msg: message to show in the exception.

    Raises:
        TypeException: if the two tensors are not in the same device.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICE(x1, x2)
    """
    if x.device != y.device:
        raise TypeError(f"Not same device for tensors. Got: {x.device} and {y.device}")


def KORNIA_CHECK_SAME_DEVICES(tensors: List[Tensor], msg: Optional[str] = None):
    """Check whether a list provided tensors live in the same device.

    Args:
        x: a list of tensors.
        msg: message to show in the exception.

    Raises:
        Exception: if all the tensors are not in the same device.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(1, 3, 1)
        >>> KORNIA_CHECK_SAME_DEVICES([x1, x2], "Tensors not in the same device")
    """
    KORNIA_CHECK(isinstance(tensors, list) and len(tensors) >= 1, "Expected a list with at least one element")
    if not all(tensors[0].device == x.device for x in tensors):
        raise Exception(f"Not same device for tensors. Got: {[x.device for x in tensors]}.\n{msg}")


def KORNIA_CHECK_SAME_SHAPE(x: Tensor, y: Tensor) -> None:
    """Check whether two tensor have the same shape.

    Args:
        x: first tensor to evaluate.
        y: sencod tensor to evaluate.
        msg: message to show in the exception.

    Raises:
        TypeException: if the two tensors have not the same shape.

    Example:
        >>> x1 = torch.rand(2, 3, 3)
        >>> x2 = torch.rand(2, 3, 3)
        >>> KORNIA_CHECK_SAME_SHAPE(x1, x2)
    """
    if x.shape != y.shape:
        raise TypeError(f"Not same shape for tensors. Got: {x.shape} and {y.shape}")


def KORNIA_CHECK_IS_COLOR(x: Tensor, msg: Optional[str] = None):
    """Check whether an image tensor is a color images.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.

    Raises:
        TypeException: if all the input tensor has not a shape :math:`(3,H,W)`.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR(img, "Image is not color")
    """
    if len(x.shape) < 3 or x.shape[-3] != 3:
        raise TypeError(f"Not a color tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_GRAY(x: Tensor, msg: Optional[str] = None):
    """Check whether an image tensor is grayscale.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.

    Raises:
        TypeException: if the tensor has not a shape :math:`(1,H,W)` or :math:`(H,W)`.

    Example:
        >>> img = torch.rand(2, 1, 4, 4)
        >>> KORNIA_CHECK_IS_GRAY(img, "Image is not grayscale")
    """
    if len(x.shape) < 2 or (len(x.shape) >= 3 and x.shape[-3] != 1):
        raise TypeError(f"Not a gray tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_IS_COLOR_OR_GRAY(x: Tensor, msg: Optional[str] = None):
    """Check whether an image tensor is grayscale or color.

    Args:
        x: image tensor to evaluate.
        msg: message to show in the exception.

    Raises:
        TypeException: if the tensor has not a shape :math:`(1,H,W)` or :math:`(3,H,W)`.

    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> KORNIA_CHECK_IS_COLOR_OR_GRAY(img, "Image is not color or grayscale")
    """
    if len(x.shape) < 3 or x.shape[-3] not in [1, 3]:
        raise TypeError(f"Not a color or gray tensor. Got: {type(x)}.\n{msg}")


def KORNIA_CHECK_DM_DESC(desc1: Tensor, desc2: Tensor, dm: Tensor):
    """Check whether the provided descriptors match with a distance matrix.

    Args:
        desc1: first descriptor tensor to evaluate.
        desc2: second descriptor tensor to evaluate.
        dm: distance matrix tensor to evaluate.

    Raises:
        TypeException: if the descriptors shape do not match with the distance matrix.

    Example:
        >>> desc1 = torch.rand(4)
        >>> desc2 = torch.rand(8)
        >>> dm = torch.rand(4, 8)
        >>> KORNIA_CHECK_DM_DESC(desc1, desc2, dm)
    """
    if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
        message = f"""distance matrix shape {dm.shape} is not
                      consistent with descriptors shape: desc1 {desc1.shape}
                      desc2 {desc2.shape}"""
        raise TypeError(message)


def KORNIA_CHECK_LAF(laf: Tensor) -> None:
    """Check whether a Local Affine Frame (laf) has a valid shape.

    Args:
        laf: local affine frame tensor to evaluate.

    Raises:
        Exception: if the input laf does not have a shape :math:`(B,N,2,3)`.

    Example:
        >>> lafs = torch.rand(2, 10, 2, 3)
        >>> KORNIA_CHECK_LAF(lafs)
    """
    KORNIA_CHECK_SHAPE(laf, ["B", "N", "2", "3"])
