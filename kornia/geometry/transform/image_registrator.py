from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, optim

from kornia.core import Module, Tensor
from kornia.geometry.conversions import angle_to_rotation_matrix, convert_affinematrix_to_homography

from .homography_warper import BaseWarper, HomographyWarper
from .pyramid import build_pyramid

__all__ = ["ImageRegistrator", "Homography", "Similarity", "BaseModel"]


class BaseModel(Module):
    @abstractmethod
    def reset_model(self) -> None: ...

    @abstractmethod
    def forward(self) -> Tensor: ...

    @abstractmethod
    def forward_inverse(self) -> Tensor: ...


class Homography(BaseModel):
    r"""Homography geometric model to be used together with ImageRegistrator module for the optimization-based
    image registration."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Parameter(torch.eye(3))
        self.reset_model()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model})"

    def reset_model(self) -> None:
        """Initializes the model with identity transform."""
        torch.nn.init.eye_(self.model)

    def forward(self) -> Tensor:
        r"""Single-batch homography".

        Returns:
            Homography matrix with shape :math:`(1, 3, 3)`.
        """
        return torch.unsqueeze(self.model / self.model[2, 2], dim=0)  # 1x3x3

    def forward_inverse(self) -> Tensor:
        r"""Interted Single-batch homography".

        Returns:
            Homography martix with shape :math:`(1, 3, 3)`.
        """
        return torch.unsqueeze(torch.inverse(self.model), dim=0)


class Similarity(BaseModel):
    """Similarity geometric model to be used together with ImageRegistrator module for the optimization-based image
    registration.

    Args:
        rotation: if True, the rotation is optimizable, else constant zero.
        scale: if True, the scale is optimizable, else constant zero.
        shift: if True, the shift is optimizable, else constant one.
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

    def forward(self) -> Tensor:
        r"""Single-batch similarity transform".

        Returns:
            Similarity with shape :math:`(1, 3, 3)`
        """
        rot = self.scale * angle_to_rotation_matrix(self.rot)
        out = convert_affinematrix_to_homography(torch.cat([rot, self.shift], dim=2))
        return out

    def forward_inverse(self) -> Tensor:
        r"""Single-batch inverse similarity transform".

        Returns:
            Similarity with shape :math:`(1, 3, 3)`
        """
        return torch.inverse(self.forward())


class ImageRegistrator(Module):
    r"""Module, which performs optimization-based image registration.

    Args:
        model_type: Geometrical model for registration. Can be string or Module.
        optimizer: optimizer class used for the optimization.
        loss_fn: torch loss function.
        pyramid_levels: number of scale pyramid levels.
        lr: learning rate for optimization.
        num_iterations: maximum number of iterations.
        tolerance: stop optimizing if loss difference is less. default 1e-4.
        warper: if model_type is not string, one needs to provide warper object.

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
        loss_fn: Callable[..., Tensor] = F.l1_loss,
        pyramid_levels: int = 5,
        lr: float = 1e-3,
        num_iterations: int = 100,
        tolerance: float = 1e-4,
        warper: Optional[Type[BaseWarper]] = None,
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

    def get_single_level_loss(self, img_src: Tensor, img_dst: Tensor, transform_model: Tensor) -> Tensor:
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
        ones = warper(torch.ones_like(img_src), transform_model)
        loss = loss.masked_select(ones > 0.9).mean()
        return loss

    def reset_model(self) -> None:
        """Calls model reset function."""
        self.model.reset_model()

    def register(
        self, src_img: Tensor, dst_img: Tensor, verbose: bool = False, output_intermediate_models: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        r"""Estimate the tranformation' which warps src_img into dst_img by gradient descent. The shape of the
        tensors is not checked, because it may depend on the model, e.g. volume registration.

        Args:
            src_img: Input image tensor.
            dst_img: Input image tensor.
            verbose: if True, outputs loss every 10 iterations.
            output_intermediate_models: if True with intermediate models

        Returns:
            the transformation between two images, shape depends on the model,
            typically [1x3x3] tensor for string model_types.
        """
        self.reset_model()
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

    def warp_src_into_dst(self, src_img: Tensor) -> Tensor:
        r"""Warp src_img with estimated model."""
        _height, _width = src_img.shape[-2:]
        warper = self.warper(_height, _width)
        img_src_to_dst = warper(src_img, self.model())
        return img_src_to_dst

    def warp_dst_inro_src(self, dst_img: Tensor) -> Tensor:
        r"""Warp src_img with inverted estimated model."""
        _height, _width = dst_img.shape[-2:]
        warper = self.warper(_height, _width)
        img_dst_to_src = warper(dst_img, self.model.forward_inverse())
        return img_dst_to_src
