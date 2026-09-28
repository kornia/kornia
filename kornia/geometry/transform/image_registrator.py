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

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, optim

from kornia.core._compat import deprecated
from kornia.core.utils import _torch_inverse_cast
from kornia.geometry.conversions import angle_to_rotation_matrix, convert_affinematrix_to_homography

from .homography_warper import BaseWarper, HomographyWarper
from .pyramid import build_pyramid

__all__ = ["BaseModel", "Homography", "ImageRegistrator", "Similarity"]


class BaseModel(nn.Module):
    """Provide an abstract base class for image registration models."""

    @abstractmethod
    def reset_model(self) -> None:
        """Reset learnable registration parameters to the identity transform."""
        ...

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Return the transform that maps destination coordinates to source coordinates.

        See :class:`ImageRegistrator` for the coordinate convention.

        Returns:
            Transform matrix tensor for the current model state. Concrete
            models return the matrix shape required by their warp function.
        """
        ...

    @abstractmethod
    def forward_inverse(self) -> torch.Tensor:
        """Return the inverse mapping for the current registration transform.

        Returns:
            Transform matrix tensor that maps source coordinates to
            destination coordinates.
        """
        ...


class Homography(BaseModel):
    r"""Homography geometric model to be used with ImageRegistrator for the optimization-based image registration."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Parameter(torch.eye(3))
        self.reset_model()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model})"

    def reset_model(self) -> None:
        """Initialize the model with identity transform."""
        torch.nn.init.eye_(self.model)

    def forward(self) -> torch.Tensor:
        r"""Single-batch homography, scaled so that its ``[2, 2]`` entry is 1.

        Returns:
            Homography matrix with shape :math:`(1, 3, 3)`.

        """
        return torch.unsqueeze(self.model / self.model[2, 2], dim=0)  # 1x3x3

    def forward_inverse(self) -> torch.Tensor:
        r"""Inverted single-batch homography.

        Returns:
            Homography matrix with shape :math:`(1, 3, 3)`.

        """
        return torch.unsqueeze(_torch_inverse_cast(self.model), dim=0)


class Similarity(BaseModel):
    """Similarity geometric model to be used with ImageRegistrator module for the optimization-based image registration.

    Convention:
        ``forward()`` is ``[[scale * R, shift], [0, 0, 1]]``, where ``R`` is
        :func:`~kornia.geometry.conversions.angle_to_rotation_matrix` of ``rot`` in degrees and ``shift`` is in the
        normalized coordinates of :class:`ImageRegistrator`.

    Args:
        rotation: if True, the rotation is optimizable, else constant zero.
        scale: if True, the scale is optimizable, else constant one.
        shift: if True, the shift is optimizable, else constant zero.

    """

    def __init__(self, rotation: bool = True, scale: bool = True, shift: bool = True) -> None:
        super().__init__()
        if rotation:
            self.rot = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("rot", torch.zeros(1))
        if shift:
            self.shift = nn.Parameter(torch.zeros(1, 2, 1))
        else:
            self.register_buffer("shift", torch.zeros(1, 2, 1))
        if scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))
        self.reset_model()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(angle = {self.rot},               \n shift={self.shift}, \n scale={self.scale})"
        )

    def reset_model(self) -> None:
        """Initialize the model with identity transform."""
        torch.nn.init.zeros_(self.rot)
        torch.nn.init.zeros_(self.shift)
        torch.nn.init.ones_(self.scale)

    def forward(self) -> torch.Tensor:
        r"""Single-batch similarity transform".

        Returns:
            Similarity with shape :math:`(1, 3, 3)`

        """
        rot = self.scale * angle_to_rotation_matrix(self.rot)
        return convert_affinematrix_to_homography(torch.cat([rot, self.shift], dim=2))

    def forward_inverse(self) -> torch.Tensor:
        r"""Single-batch inverse similarity transform".

        Returns:
            Similarity with shape :math:`(1, 3, 3)`

        """
        return _torch_inverse_cast(self.forward())


class ImageRegistrator(nn.Module):
    r"""nn.Module, which performs optimization-based image registration.

    Convention:
        - :meth:`register` returns ``self.model()``, which maps **destination** coordinates to **source**
          coordinates, normalized to :math:`[-1, 1]` with ``align_corners=False``: the ``src_homo_dst`` argument of
          :func:`~kornia.geometry.transform.homography_warp`. :meth:`warp_src_into_dst` warps with it, and
          :meth:`warp_dst_into_src` with ``self.model.forward_inverse()``, which maps source to destination.
        - ``denormalize_homography(M, (H, W), (H, W), align_corners=False)`` converts the returned ``M`` to pixels:
          content that moves by :math:`(t_x, t_y)` pixels from ``src_img`` to ``dst_img`` gives a shift of
          :math:`(-t_x, -t_y)`.
        - :meth:`register` always starts from the identity, so a loaded ``state_dict`` drives the warps but is not a
          warm start.

    Args:
        model_type: Geometrical model for registration: ``'homography'``, ``'similarity'``, or ``'translation'``,
            ``'rotation'`` or ``'scale'`` (a :class:`Similarity` that optimizes only that parameter), or a
            :class:`BaseModel` instance together with ``warper``.
        optimizer: optimizer class used for the optimization.
        loss_fn: torch loss function, called with ``reduction='none'``.
        pyramid_levels: number of scale pyramid levels.
        lr: learning rate for optimization.
        num_iterations: maximum number of iterations at each pyramid level, from coarse to fine.
        tolerance: stop optimizing a pyramid level if the loss changes by less. default 1e-4.
        warper: the warper class, called as ``warper(height, width)``. Required when ``model_type`` is a module;
            a string ``model_type`` uses :class:`~kornia.geometry.transform.HomographyWarper`.
        allow_shape_mismatch: if True, :meth:`register` resizes ``src_img`` bilinearly to the height and width of
            ``dst_img``; if False, images of different shapes raise ``ValueError``.

    Example:
        >>> from kornia.geometry import ImageRegistrator
        >>> img_src = torch.rand(1, 1, 32, 32)
        >>> img_dst = torch.rand(1, 1, 32, 32)
        >>> registrator = ImageRegistrator('similarity')
        >>> homo = registrator.register(img_src, img_dst)

    """

    # TODO: resolve better type, potentially using factory.
    def __init__(
        self,
        model_type: Union[str, BaseModel] = "homography",
        optimizer: Type[optim.Optimizer] = optim.Adam,
        loss_fn: Callable[..., torch.Tensor] = F.l1_loss,
        pyramid_levels: int = 5,
        lr: float = 1e-3,
        num_iterations: int = 100,
        tolerance: float = 1e-4,
        warper: Optional[Type[BaseWarper]] = None,
        allow_shape_mismatch: bool = False,
    ) -> None:
        super().__init__()
        self.known_models = ["homography", "similarity", "translation", "scale", "rotation"]
        # We provide pre-defined combinations or allow user to supply model
        # together with warper
        if not isinstance(model_type, str):
            if warper is None:
                raise ValueError("You must supply warper together with custom model")
            self.warper = warper
            self.model = model_type
        elif model_type.lower() == "homography":
            self.warper = HomographyWarper
            self.model = Homography()
        elif model_type.lower() == "similarity":
            self.warper = HomographyWarper
            self.model = Similarity(True, True, True)
        elif model_type.lower() == "translation":
            self.warper = HomographyWarper
            self.model = Similarity(False, False, True)
        elif model_type.lower() == "rotation":
            self.warper = HomographyWarper
            self.model = Similarity(True, False, False)
        elif model_type.lower() == "scale":
            self.warper = HomographyWarper
            self.model = Similarity(False, True, False)
        else:
            raise ValueError(f"{model_type} is not supported. Try {self.known_models}")
        self.pyramid_levels = pyramid_levels
        self.optimizer = optimizer
        self.lr = lr
        self.loss_fn = loss_fn
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.allow_shape_mismatch = allow_shape_mismatch

    def get_single_level_loss(
        self, img_src: torch.Tensor, img_dst: torch.Tensor, transform_model: torch.Tensor
    ) -> torch.Tensor:
        """Warp img_src into img_dst with transform_model and returns loss."""
        # ToDo: Make possible registration of images of different shape
        if img_src.shape != img_dst.shape:
            raise ValueError(
                "Cannot register images of different shapes                             "
                f" {img_src.shape} {img_dst.shape:} "
            )
        _height, _width = img_dst.shape[-2:]
        warper = self.warper(_height, _width)
        img_src_to_dst = warper(img_src, transform_model)
        # compute and mask loss
        loss = self.loss_fn(img_src_to_dst, img_dst, reduction="none")  # 1xCxHxW
        ones_tensor = warper(torch.ones_like(img_src), transform_model)
        return loss.masked_select(ones_tensor > 0.9).mean()

    def reset_model(self) -> None:
        """Call model reset function."""
        self.model.reset_model()

    def register(
        self,
        src_img: torch.Tensor,
        dst_img: torch.Tensor,
        verbose: bool = False,
        output_intermediate_models: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""Estimate the transformation which warps src_img into dst_img by gradient descent.

        Args:
            src_img: Input image torch.Tensor.
            dst_img: Input image torch.Tensor.
            verbose: if True, outputs loss every 10 iterations.
            output_intermediate_models: if True with intermediate models

        Returns:
            the transformation between two images, shape depends on the model,
            typically [1x3x3] torch.Tensor for string model_types.

        """
        self.reset_model()
        if src_img.shape != dst_img.shape:
            if not self.allow_shape_mismatch:
                raise ValueError(
                    f"Cannot register images of different shapes {src_img.shape} {dst_img.shape}. "
                    "Consider setting `allow_shape_mismatch = True`"
                )
            src_img = F.interpolate(
                src_img,
                size=dst_img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        # ToDo: better parameter passing to optimizer
        _opt_args: Dict[str, Any] = {}
        _opt_args["lr"] = self.lr
        opt = self.optimizer(self.model.parameters(), **_opt_args)

        # compute the gaussian pyramids
        # [::-1] because we have to register from coarse to fine
        img_src_pyr = build_pyramid(src_img, self.pyramid_levels)[::-1]
        img_dst_pyr = build_pyramid(dst_img, self.pyramid_levels)[::-1]
        prev_loss = 1e10
        aux_models = []
        if len(img_dst_pyr) != len(img_src_pyr):
            raise ValueError("Cannot register images of different sizes")
        for img_src_level, img_dst_level in zip(img_src_pyr, img_dst_pyr):
            for i in range(self.num_iterations):
                # compute gradient and update optimizer parameters
                opt.zero_grad()
                loss = self.get_single_level_loss(img_src_level, img_dst_level, self.model())
                loss += self.get_single_level_loss(img_dst_level, img_src_level, self.model.forward_inverse())
                current_loss = loss.item()
                if abs(current_loss - prev_loss) < self.tolerance:
                    break
                prev_loss = current_loss
                loss.backward()
                if verbose and (i % 10 == 0):
                    print(f"Loss = {current_loss:.4f}, iter={i}")
                opt.step()
            if output_intermediate_models:
                aux_models.append(self.model().clone().detach())
        if output_intermediate_models:
            return self.model(), aux_models
        return self.model()

    def warp_src_into_dst(self, src_img: torch.Tensor) -> torch.Tensor:
        r"""Warp src_img with estimated model."""
        _height, _width = src_img.shape[-2:]
        warper = self.warper(_height, _width)
        return warper(src_img, self.model())

    def warp_dst_into_src(self, dst_img: torch.Tensor) -> torch.Tensor:
        r"""Warp dst_img with inverted estimated model."""
        _height, _width = dst_img.shape[-2:]
        warper = self.warper(_height, _width)
        return warper(dst_img, self.model.forward_inverse())

    @deprecated(replace_with="ImageRegistrator.warp_dst_into_src", version="0.9.0")
    def warp_dst_inro_src(self, dst_img: torch.Tensor) -> torch.Tensor:
        r"""Deprecated alias for :meth:`warp_dst_into_src`."""
        return self.warp_dst_into_src(dst_img)
