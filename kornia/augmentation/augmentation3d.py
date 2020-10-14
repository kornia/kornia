from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
from torch.nn.functional import pad

from kornia.constants import Resample, BorderType
from .base import AugmentationBase3D
from . import functional as F
from . import random_generator as rg
from .utils import (
    _range_bound,
    _tuple_range_reader,
    _singular_range_check
)


class RandomHorizontalFlip3D(AugmentationBase3D):
    r"""Applies a random horizontal flip to a 3D image tensor with a given probability.

    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> x = torch.eye(3).repeat(3, 1, 1)
        >>> seq = RandomHorizontalFlip3D(p=1.0, return_transform=True)
        >>> seq(x)
        (tensor([[[[[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]]]]]), tensor([[[-1.,  0.,  0.,  2.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]]]))

    """

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5) -> None:
        super(RandomHorizontalFlip3D, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch)

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_hflip_transformation3d(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_hflip3d(input)


class RandomVerticalFlip3D(AugmentationBase3D):
    r"""Applies a random vertical flip to a 3D image tensor with a given probability.

    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Vertically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> x = torch.eye(3).repeat(3, 1, 1)
        >>> seq = RandomVerticalFlip3D(p=1.0, return_transform=True)
        >>> seq(x)
        (tensor([[[[[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]]]]]), tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0., -1.,  0.,  2.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]]]))
    """

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5) -> None:
        super(RandomVerticalFlip3D, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch)

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_vflip_transformation3d(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_vflip3d(input)


class RandomDepthicalFlip3D(AugmentationBase3D):
    r"""Applies a random flip along the depth axis of a 3D image tensor with a given probability.

    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Depthically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5.
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> x = torch.eye(3).repeat(3, 1, 1)
        >>> seq = RandomDepthicalFlip3D(p=1.0, return_transform=True)
        >>> seq(x)
        (tensor([[[[[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]],
        <BLANKLINE>
                  [[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]],
        <BLANKLINE>
                  [[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]]]]]), tensor([[[ 1.,  0.,  0.,  0.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0., -1.,  2.],
                 [ 0.,  0.,  0.,  1.]]]))

    """

    def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5) -> None:
        super(RandomDepthicalFlip3D, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch)

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_dflip_transformation3d(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_dflip3d(input)


class RandomAffine3D(AugmentationBase3D):
    r"""Applies a 3D affine transformation to a 3D image tensor with a given probability.

    The transformation is computed so that the center is kept invariant.

    Args:
        degrees (float or tuple or list): Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal, vertical and
        depthical translations (dx,dy,dz). For example translate=(a, b, c), then
            horizontal shift will be randomly sampled in the range -img_width * a < dx < img_width * a
            vertical shift will be randomly sampled in the range -img_height * b < dy < img_height * b.
            depthical shift will be randomly sampled in the range -img_depth * c < dz < img_depth * c.
            Will not translate by default.
        scale (tuple, optional): scaling factor interval.
            If (a, b) represents isotropic scaling, the scale is randomly sampled from the range a <= scale <= b.
            If ((a, b), (c, d), (e, f)), the scale is randomly sampled from the range a <= scale_x <= b,
            c <= scale_y <= d, e <= scale_z <= f. Will keep original scale by default.
        shear (sequence or float, optional): Range of degrees to select from.
            If shear is a number, a shear to the 6 facets in the range (-shear, +shear) will be apllied.
            If shear is a tuple of 2 values, a shear to the 6 facets in the range (shear[0], shear[1]) will be applied.
            If shear is a tuple of 6 values, a shear to the i-th facet in the range (-shear[i], shear[i])
            will be applied.
            If shear is a tuple of 6 tuples, a shear to the i-th facet in the range (-shear[i, 0], shear[i, 1])
            will be applied.
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR.
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomAffine3D((15., 20., 20.), p=1., return_transform=True)
        >>> aug(input)
        (tensor([[[[[0.4503, 0.4763, 0.1680],
                   [0.2029, 0.4267, 0.3515],
                   [0.3195, 0.5436, 0.3706]],
        <BLANKLINE>
                  [[0.5255, 0.3508, 0.4858],
                   [0.0795, 0.1689, 0.4220],
                   [0.5306, 0.7234, 0.6879]],
        <BLANKLINE>
                  [[0.2971, 0.2746, 0.3471],
                   [0.4924, 0.4960, 0.6460],
                   [0.3187, 0.4556, 0.7596]]]]]), tensor([[[ 0.9722, -0.0603,  0.2262, -0.1381],
                 [ 0.1131,  0.9669, -0.2286,  0.1486],
                 [-0.2049,  0.2478,  0.9469,  0.0102],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]))
    """

    def __init__(
        self, degrees: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float],
                             Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float],
                              Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]] = None,
        shears: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float, float, float, float],
                      Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float],
                            Tuple[float, float], Tuple[float, float]]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False, p: float = 0.5
    ) -> None:
        super(RandomAffine3D, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)
        self.degrees = _tuple_range_reader(degrees, 3)
        self.shear: Optional[torch.Tensor] = None
        if shears is not None:
            self.shear = _tuple_range_reader(shears, 6)

        # check translation range
        self.translate: Optional[torch.Tensor] = None
        if translate is not None:
            self.translate = translate if isinstance(translate, torch.Tensor) else torch.tensor(translate)
            _singular_range_check(self.translate, 'translate', bounds=(0, 1), mode='3d')

        # check scale range
        self.scale: Optional[torch.Tensor] = None
        if scale is not None:
            self.scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
            if self.scale.shape == torch.Size([2]):
                self.scale = self.scale.unsqueeze(0).repeat(3, 1)
            elif self.scale.shape != torch.Size([3, 2]):
                raise ValueError("'scale' shall be either shape (2) or (3, 2). Got {self.scale}")
            _singular_range_check(self.scale[0], 'scale-x', bounds=(0, float('inf')), mode='2d')
            _singular_range_check(self.scale[1], 'scale-y', bounds=(0, float('inf')), mode='2d')
            _singular_range_check(self.scale[2], 'scale-z', bounds=(0, float('inf')), mode='2d')

        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            resample=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = (f"(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, "
                f"resample={self.resample.name}, align_corners={self.align_corners}")
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_affine_generator3d(
            batch_shape[0], batch_shape[-3], batch_shape[-2], batch_shape[-1], self.degrees,
            self.translate, self.scale, self.shear, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_affine_transformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_affine3d(input, params, self.flags)


class RandomRotation3D(AugmentationBase3D):

    r"""Applies a random rotation to a 3D tensor image or a batch of tensor images given an amount of degrees.

    Input should be a tensor of shape (C, D, H, W) or a batch of tensors :math:`(B, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will rotate the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        degrees (float or tuple or list): Range of degrees to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        align_corners(bool): interpolation flag. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomRotation3D((15., 20., 20.), p=1.0, return_transform=True)
        >>> aug(input)
        (tensor([[[[[0.4963, 0.5013, 0.2314],
                   [0.1015, 0.3624, 0.4779],
                   [0.2669, 0.5749, 0.4081]],
        <BLANKLINE>
                  [[0.4426, 0.4198, 0.2713],
                   [0.2911, 0.1689, 0.4538],
                   [0.3939, 0.6022, 0.6166]],
        <BLANKLINE>
                  [[0.1496, 0.3740, 0.3151],
                   [0.4169, 0.4803, 0.5804],
                   [0.3856, 0.4253, 0.9527]]]]]), tensor([[[ 0.9722,  0.1131, -0.2049,  0.1196],
                 [-0.0603,  0.9669,  0.2478, -0.1545],
                 [ 0.2262, -0.2286,  0.9469,  0.0556],
                 [ 0.0000,  0.0000,  0.0000,  1.0000]]]))
    """

    def __init__(
        self, degrees: Union[torch.Tensor, float, Tuple[float, float, float],
                             Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
        interpolation: Optional[Union[str, int, Resample]] = None,
        resample: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False, p: float = 0.5
    ) -> None:
        super(RandomRotation3D, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)
        self.degrees = _tuple_range_reader(degrees, 3)
        if interpolation is not None:
            import warnings
            warnings.warn("interpolation is deprecated. Please use resample instead.", category=DeprecationWarning)
            self.resample = Resample.get(interpolation)

        self.resample = Resample.get(resample)
        self.align_corners = align_corners
        self.flags: Dict[str, torch.Tensor] = dict(
            resample=torch.tensor(self.resample.value),
            align_corners=torch.tensor(align_corners)
        )

    def __repr__(self) -> str:
        repr = f"(degrees={self.degrees}, resample={self.resample.name}, align_corners={self.align_corners}"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_rotation_generator3d(batch_shape[0], self.degrees, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_rotate_tranformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_rotation3d(input, params, self.flags)


class RandomMotionBlur3D(AugmentationBase3D):
    r"""Perform motion blur on 3D volumes (5D tensor).

    Args:
        p (float): probability of applying the transformation. Default value is 0.5.
        kernel_size (int or Tuple[int, int]): motion kernel size (odd and positive).
            If int, the kernel will have a fixed size.
            If Tuple[int, int], it will randomly generate the value from the range batch-wisely.
        angle (float or tuple or list): Range of degrees to select from.
            If angle is a number, then yaw, pitch, roll will be generated from the range of (-angle, +angle).
            If angle is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If angle is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If angle is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        direction (float or Tuple[float, float]): forward/backward direction of the motion blur.
            Lower values towards -1.0 will point the motion blur towards the back (with angle provided via angle),
            while higher values towards 1.0 will point the motion blur forward. A value of 0.0 leads to a
            uniformly (but still angled) motion blur.
            If float, it will generate the value from (-direction, direction).
            If Tuple[int, int], it will randomly generate the value from the range.
        border_type (int, str or kornia.BorderType): the padding mode to be applied before convolving.
            CONSTANT = 0, REFLECT = 1, REPLICATE = 2, CIRCULAR = 3. Default: BorderType.CONSTANT.

    Shape:
        - Input: :math:`(B, C, D, H, W)`
        - Output: :math:`(B, C, D, H, W)`

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 5, 5)
        >>> motion_blur = RandomMotionBlur3D(3, 35., 0.5, p=1.)
        >>> motion_blur(input)
        tensor([[[[[ 0.1654,  5.0626,  3.7446,  6.8393,  3.5563],
                   [-5.0999,  2.1964,  2.4311, -1.8043,  1.4908],
                   [-3.2720, -2.1332,  2.1459,  1.7491,  1.3186],
                   [-2.6045,  5.2450,  6.0367,  1.0320,  6.7205],
                   [-4.5962, -5.2259, -0.9809, -1.6601,  0.1398]],
        <BLANKLINE>
                  [[ 0.1843,  2.6303,  7.0769,  1.3952,  2.1699],
                   [-6.4852,  0.3011,  0.3557,  4.5564,  5.5316],
                   [-6.4329,  2.8427,  0.2914,  3.6307,  0.4988],
                   [-1.2635, -4.5109,  1.3771,  0.8912,  2.1555],
                   [-1.6038, -3.7698,  0.0352, -0.8663,  0.1607]],
        <BLANKLINE>
                  [[ 0.2733,  1.8909,  4.7292,  1.0408,  1.4417],
                   [-6.8248,  0.8740,  1.4280, -4.1784,  5.0158],
                   [-3.8589,  2.7687,  2.9419,  3.2143,  6.3996],
                   [-4.7014, -0.3668, -1.8947,  0.1953,  7.2572],
                   [-3.5506, -4.1522, -5.3692, -6.4713,  0.3117]]]]])
    """

    def __init__(
            self, kernel_size: Union[int, Tuple[int, int]],
            angle: Union[torch.Tensor, float, Tuple[float, float]],
            direction: Union[torch.Tensor, float, Tuple[float, float]],
            border_type: Union[int, str, BorderType] = BorderType.CONSTANT.name,
            return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5
    ) -> None:
        super(RandomMotionBlur3D, self).__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.)
        self.kernel_size: Union[int, Tuple[int, int]] = kernel_size

        self.angle: torch.Tensor = _tuple_range_reader(angle, 3)

        direction = \
            cast(torch.Tensor, direction) if isinstance(direction, torch.Tensor) else torch.tensor(direction)
        self.direction = _range_bound(direction, 'direction', center=0., bounds=(-1, 1))
        self.border_type = BorderType.get(border_type)
        self.flags: Dict[str, torch.Tensor] = {
            "border_type": torch.tensor(self.border_type.value)
        }

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, angle={self.angle}, direction={self.direction}, " +\
            f"border_type='{self.border_type.name.lower()}'"
        return self.__class__.__name__ + f"({repr}, {super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_motion_blur_generator3d(
            batch_shape[0], self.kernel_size, self.angle, self.direction, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation3d(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_motion_blur3d(input, params, self.flags)


class RandomEqualize3D(AugmentationBase3D):
    r"""Random 3D equalization of a volumetric image.

    Args:
        p (float): probability of the image being equalized. Default value is 0.5.

        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomEqualize3D(p=1.0)
        >>> aug(input)
        tensor([[[[[0.4963, 0.7682, 0.0885],
                   [0.1320, 0.3074, 0.6341],
                   [0.4901, 0.8964, 0.4556]],
        <BLANKLINE>
                  [[0.6323, 0.3489, 0.4017],
                   [0.0223, 0.1689, 0.2939],
                   [0.5185, 0.6977, 0.8000]],
        <BLANKLINE>
                  [[0.1610, 0.2823, 0.6816],
                   [0.9152, 0.3971, 0.8742],
                   [0.4194, 0.5529, 0.9527]]]]])
    """

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomEqualize3D, self).__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return dict()

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_intensity_transformation3d(input)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_equalize3d(input, params)
