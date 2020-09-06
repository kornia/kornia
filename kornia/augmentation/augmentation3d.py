from typing import Callable, Tuple, Union, List, Optional, Dict, cast

import torch
import torch.nn as nn
from torch.nn.functional import pad

from kornia.constants import Resample, BorderType
from kornia.augmentation.augmentation import AugmentationBase
from kornia.augmentation import functional3d as F
from kornia.augmentation import random_generator as rg
from kornia.augmentation import random_generator3d as rg3
from kornia.augmentation.utils import (
    _infer_batch_shape3d,
    _tuple_range_reader,
    _singular_range_check
)


class AugmentationBase3D(AugmentationBase):
    def __init__(self, return_transform: bool = False) -> None:
        super(AugmentationBase3D, self).__init__(return_transform=return_transform)

    def infer_batch_shape(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Size:
        return _infer_batch_shape3d(input)


class RandomHorizontalFlip3D(AugmentationBase3D):

    r"""Horizontally flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

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

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomHorizontalFlip3D, self).__init__(return_transform)
        self.p: float = p
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_hflip_transformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_hflip3d(input, params)


class RandomVerticalFlip3D(AugmentationBase3D):

    r"""Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Vertically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

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

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomVerticalFlip3D, self).__init__(return_transform)
        self.p: float = p
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_vflip_transformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_vflip3d(input, params)


class RandomDepthicalFlip3D(AugmentationBase3D):

    r"""Depthically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape :math:`(C, D, H, W)` or a batch of tensors :math:`(*, C, D, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Depthically flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False

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

    def __init__(self, p: float = 0.5, return_transform: bool = False, same_on_batch: bool = False) -> None:
        super(RandomDepthicalFlip3D, self).__init__(return_transform)
        self.p: float = p
        self.same_on_batch = same_on_batch

    def __repr__(self) -> str:
        repr = f"(p={self.p}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch})"
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg.random_prob_generator(batch_shape[0], self.p, self.same_on_batch)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_dflip_transformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_dflip3d(input, params)


class RandomAffine3D(AugmentationBase3D):
    r"""Random 3D affine transformation of the image keeping center invariant.

    Args:
        degrees (float or tuple or list): Range of yaw (x-axis), pitch (y-axis), roll (z-axis) to select from.
            If degrees is a number, then yaw, pitch, roll will be generated from the range of (-degrees, +degrees).
            If degrees is a tuple of (min, max), then yaw, pitch, roll will be generated from the range of (min, max).
            If degrees is a list of floats [a, b, c], then yaw, pitch, roll will be generated from (-a, a), (-b, b)
            and (-c, c).
            If degrees is a list of tuple ((a, b), (m, n), (x, y)), then yaw, pitch, roll will be generated from
            (a, b), (m, n) and (x, y).
            Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for depthical, horizontal
            and vertical translations. For example translate=(a, b, c), then
            depthical shift will be randomly sampled in the range -img_depth * a < dx < img_depth * a
            horizontal shift will be randomly sampled in the range -img_width * b < dy < img_width * b.
            vertical shift will be randomly sampled in the range -img_height * b < dy < img_height * b.
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
        resample (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation
            applied to each. Default: False.
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomAffine3D((15., 20., 20.), return_transform=True)
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
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False
    ) -> None:
        super(RandomAffine3D, self).__init__(return_transform)
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

        self.resample: Resample = Resample.get(resample)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) -> str:
        repr = f"(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, "
        f"resample={self.resample.name}, return_transform={self.return_transform}, same_on_batch={self.same_on_batch} "
        f"align_corners={self.align_corners}"
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg3.random_affine_generator3d(
            batch_shape[0], batch_shape[-3], batch_shape[-2], batch_shape[-1], self.degrees,
            self.translate, self.scale, self.shear, self.resample, self.same_on_batch, self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_affine_transformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_affine3d(input, params)


class RandomRotation3D(AugmentationBase3D):

    r"""Rotate a tensor image or a batch of tensor images a random amount of degrees.
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
        interpolation (int, str or kornia.Resample): Default: Resample.BILINEAR
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated
        same_on_batch (bool): apply the same transformation across the batch. Default: False
        align_corners(bool): interpolation flag. Default: False.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> input = torch.rand(1, 1, 3, 3, 3)
        >>> aug = RandomRotation3D((15., 20., 20.), return_transform=True)
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
        interpolation: Union[str, int, Resample] = Resample.BILINEAR.name,
        return_transform: bool = False, same_on_batch: bool = False, align_corners: bool = False
    ) -> None:
        super(RandomRotation3D, self).__init__(return_transform)
        self.degrees = _tuple_range_reader(degrees, 3)
        self.interpolation: Resample = Resample.get(interpolation)
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def __repr__(self) -> str:
        repr = f"(degrees={self.degrees}, interpolation={self.interpolation.name}, "
        f"return_transform={self.return_transform}, same_on_batch={self.same_on_batch}), "
        f"align_corners={self.align_corners}"
        return self.__class__.__name__ + repr

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return rg3.random_rotation_generator3d(batch_shape[0], self.degrees, self.interpolation,
                                               self.same_on_batch, self.align_corners)

    def compute_transformation(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.compute_rotate_tranformation3d(input, params)

    def apply_transform(self, input: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.apply_rotation3d(input, params)
