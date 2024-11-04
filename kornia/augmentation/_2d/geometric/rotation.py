from typing import Any, Dict, List, Optional, Tuple, Union

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.constants import Resample
from kornia.core import Tensor, as_tensor
from kornia.geometry.transform import affine
from kornia.geometry.transform.affwarp import _compute_rotation_matrix, _compute_tensor_center
from kornia.utils.misc import eye_like


class RandomRotation(GeometricAugmentationBase2D):
    r"""Apply a random rotation to a tensor image or a batch of tensor images given an amount of degrees.

    .. image:: _static/img/RandomRotation.png

    Args:
        degrees: range of degrees to select from. If degrees is a number the
          range of degrees to select from will be (-degrees, +degrees).
        resample: Default: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.affine`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.tensor([[1., 0., 0., 2.],
        ...                       [0., 0., 0., 0.],
        ...                       [0., 1., 2., 0.],
        ...                       [0., 0., 1., 2.]])
        >>> aug = RandomRotation(degrees=45.0, p=1.)
        >>> out = aug(input)
        >>> out
        tensor([[[[0.9824, 0.0088, 0.0000, 1.9649],
                  [0.0000, 0.0029, 0.0000, 0.0176],
                  [0.0029, 1.0000, 1.9883, 0.0000],
                  [0.0000, 0.0088, 1.0117, 1.9649]]]])
        >>> aug.transform_matrix
        tensor([[[ 1.0000, -0.0059,  0.0088],
                 [ 0.0059,  1.0000, -0.0088],
                 [ 0.0000,  0.0000,  1.0000]]])
        >>> inv = aug.inverse(out)

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomRotation(degrees=45.0, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    # Note: Extra params, center=None, fill=0 in TorchVision

    def __init__(
        self,
        degrees: Union[Tensor, float, Tuple[float, float], List[float]],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((degrees, "degrees", 0.0, (-360.0, 360.0)))

        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}

    def compute_transformation(self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any]) -> Tensor:
        # TODO: Update to use `get_rotation_matrix2d`
        angles: Tensor = params["degrees"].to(input)

        center: Tensor = _compute_tensor_center(input)
        rotation_mat: Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: Tensor = eye_like(3, input, shared_memory=False)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]

        return trans_mat

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return affine(input, transform[..., :2, :3], flags["resample"].name.lower(), "zeros", flags["align_corners"])

    def inverse_transform(
        self,
        input: Tensor,
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )


class RandomRotation90(GeometricAugmentationBase2D):
    r"""Apply a random 90 * n degree rotation to a tensor image or a batch of tensor images.

    Args:
        times: the range of n times 90 degree rotation needs to be applied.
        resample: Default: the interpolation mode.
        same_on_batch: apply the same transformation across the batch.
        align_corners: interpolation flag.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.geometry.transform.affine`. This version is relatively
        slow as it operates based on affine transformations.

    Examples:
        >>> rng = torch.manual_seed(1)
        >>> torch.set_printoptions(sci_mode=False)
        >>> input = torch.tensor([[1., 0., 0., 2.],
        ...                       [0., 0., 0., 0.],
        ...                       [0., 1., 2., 0.],
        ...                       [0., 0., 1., 2.]])
        >>> aug = RandomRotation90(times=(1, 1), p=1.)
        >>> out = aug(input)
        >>> out
        tensor([[[[    2.0000,     0.0000,     0.0000,     2.0000],
                  [    0.0000,     0.0000,     2.0000,     1.0000],
                  [    0.0000,     0.0000,     1.0000,     0.0000],
                  [    1.0000,     0.0000,     0.0000,     0.0000]]]])
        >>> aug.transform_matrix
        tensor([[[    -0.0000,      1.0000,      0.0000],
                 [    -1.0000,     -0.0000,      3.0000],
                 [     0.0000,      0.0000,      1.0000]]])
        >>> inv = aug.inverse(out)
        >>> torch.set_printoptions(profile='default')

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomRotation90(times=(-1, 1), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        times: tuple[int, int],
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = True,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((times, "times", 0.0, (-3, 3)))

        self.flags = {"resample": Resample.get(resample), "align_corners": align_corners}

    def compute_transformation(self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]) -> Tensor:
        # TODO: Update to use `get_rotation_matrix2d`
        angles: Tensor = 90.0 * params["times"].round().to(input)

        center: Tensor = _compute_tensor_center(input)
        rotation_mat: Tensor = _compute_rotation_matrix(angles, center.expand(angles.shape[0], -1))

        # rotation_mat is B x 2 x 3 and we need a B x 3 x 3 matrix
        trans_mat: Tensor = eye_like(3, input, shared_memory=False)
        trans_mat[:, 0] = rotation_mat[:, 0]
        trans_mat[:, 1] = rotation_mat[:, 1]

        return trans_mat

    def apply_transform(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")

        return affine(input, transform[..., :2, :3], flags["resample"].name.lower(), "zeros", flags["align_corners"])

    def inverse_transform(
        self,
        input: Tensor,
        flags: dict[str, Any],
        transform: Optional[Tensor] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        if not isinstance(transform, Tensor):
            raise TypeError(f"Expected the `transform` be a Tensor. Got {type(transform)}.")
        return self.apply_transform(
            input,
            params=self._params,
            transform=as_tensor(transform, device=input.device, dtype=input.dtype),
            flags=flags,
        )
