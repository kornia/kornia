from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.enhance.normalize import Normalize
from kornia.utils.helpers import map_location_to_cpu

from .dedode_models import DeDoDeDescriptor, DeDoDeDetector, get_descriptor, get_detector
from .utils import dedode_denormalize_pixel_coordinates, sample_keypoints

urls: Dict[str, Dict[str, str]] = {
    "detector": {
        "L-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
        "L-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth",
        "L-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth",
        "L-C4-v2": "https://github.com/Parskatt/DeDoDe/releases/download/v2/dedode_detector_L_v2.pth",
    },
    "descriptor": {
        "B-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
        "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
        "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
        "G-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
        "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth",
        "G-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_SO2_Spread_descriptor_setting_C.pth",
    },
}


class DeDoDe(Module):
    r"""Module which detects and/or describes local features in an image using the DeDode method.

    See :cite:`edstedt2024dedode` for details.

    .. note:: DeDode takes ImageNet normalized images as input (not in range [0, 1]).

    Args:
        detector_model: The detector model kind. Available options are: `L`.
        descriptor_model: The descriptor model kind. Available options are: `G` or `B`
        amp_dtype: The automatic mixed precision desired.

    Example:
        >>> dedode = DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright")
        >>> images = torch.randn(1, 3, 256, 256)
        >>> keypoints, scores = dedode.detect(images)
        >>> descriptions = dedode.describe(images, keypoints = keypoints)
        >>> keypoints, scores, features = dedode(images) # alternatively do both
    """

    # TODO: implement steerers and mnn matchers
    def __init__(
        self,
        detector_model: Literal["L"] = "L",
        descriptor_model: Literal["G", "B"] = "G",
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.detector: DeDoDeDetector = get_detector(detector_model, amp_dtype)
        self.descriptor: DeDoDeDescriptor = get_descriptor(descriptor_model, amp_dtype)
        self.normalizer = Normalize(torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

    def forward(
        self,
        images: Tensor,
        n: Optional[int] = 10_000,
        apply_imagenet_normalization: bool = True,
        pad_if_not_divisible: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Detects and describes keypoints in the input images.

        Args:
            images: A tensor of shape :math:`(B, 3, H, W)` containing the ImageNet-Normalized input images.
            n: The number of keypoints to detect.
            apply_imagenet_normalization: Whether to apply ImageNet normalization to the input images.

        Returns:
            keypoints: A tensor of shape :math:`(B, N, 2)` containing the detected keypoints in the image range,
            unlike `.detect()` function
            scores: A tensor of shape :math:`(B, N)` containing the scores of the detected keypoints.
            descriptions: A tensor of shape :math:`(B, N, DIM)` containing the descriptions of the detected keypoints.
            DIM is 256 for B and 512 for G.
        """
        if apply_imagenet_normalization:
            images = self.normalizer(images)
        B, C, H, W = images.shape
        if pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 14 - h % 14 if h % 14 > 0 else 0
            pd_w = 14 - w % 14 if w % 14 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)
        keypoints, scores = self.detect(images, n=n, apply_imagenet_normalization=False, crop_h=h, crop_w=w)
        descriptions = self.describe(images, keypoints, apply_imagenet_normalization=False)
        return dedode_denormalize_pixel_coordinates(keypoints, H, W), scores, descriptions

    @torch.inference_mode()
    def detect(
        self,
        images: Tensor,
        n: Optional[int] = 10_000,
        apply_imagenet_normalization: bool = True,
        pad_if_not_divisible: bool = True,
        crop_h: Optional[int] = None,
        crop_w: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Detects keypoints in the input images.

        Args:
            images: A tensor of shape :math:`(B, 3, H, W)` containing the input images.
            n: The number of keypoints to detect.
            apply_imagenet_normalization: Whether to apply ImageNet normalization to the input images.
            crop_h: The height of the crop to be used for detection. If None, the full image is used.
            crop_w: The width of the crop to be used for detection. If None, the full image is used.

        Returns:
            keypoints: A tensor of shape :math:`(B, N, 2)` containing the detected keypoints,
            normalized to the range :math:`[-1, 1]`.
            scores: A tensor of shape :math:`(B, N)` containing the scores of the detected keypoints.
        """
        KORNIA_CHECK_SHAPE(images, ["B", "3", "H", "W"])
        self.train(False)
        if pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 14 - h % 14 if h % 14 > 0 else 0
            pd_w = 14 - w % 14 if w % 14 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)
        if apply_imagenet_normalization:
            images = self.normalizer(images)
        B, C, H, W = images.shape
        logits = self.detector.forward(images)
        if crop_h is not None and crop_w is not None:
            logits = logits[..., :crop_h, :crop_w]
            H, W = crop_h, crop_w
        scoremap = logits.reshape(B, H * W).softmax(dim=-1).reshape(B, H, W)
        keypoints, confidence = sample_keypoints(scoremap, num_samples=n)
        return keypoints, confidence

    @torch.inference_mode()
    def describe(
        self, images: Tensor, keypoints: Optional[Tensor] = None, apply_imagenet_normalization: bool = True
    ) -> Tensor:
        """Describes keypoints in the input images. If keypoints are not provided, returns the dense descriptors.

        Args:
            images: A tensor of shape :math:`(B, 3, H, W)` containing the input images.
            keypoints: An optional tensor of shape :math:`(B, N, 2)` containing the detected keypoints.
            apply_imagenet_normalization: Whether to apply ImageNet normalization to the input images.

        Returns:
            descriptions: A tensor of shape :math:`(B, N, DIM)` containing the descriptions of the detected keypoints.
            If the dense descriptors are requested, the shape is :math:`(B, DIM, H, W)`.
        """
        KORNIA_CHECK_SHAPE(images, ["B", "3", "H", "W"])
        B, C, H, W = images.shape
        if keypoints is not None:
            KORNIA_CHECK_SHAPE(keypoints, ["B", "N", "2"])
        if apply_imagenet_normalization:
            images = self.normalizer(images)
        self.train(False)
        descriptions = self.descriptor.forward(images)
        if keypoints is not None:
            described_keypoints = F.grid_sample(
                descriptions.float(), keypoints[:, None], mode="bilinear", align_corners=False
            )[:, :, 0].mT
            return described_keypoints
        return descriptions

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "L-C4-v2",
        descriptor_weights: str = "G-upright",
        amp_dtype: torch.dtype = torch.float16,
    ) -> Module:
        r"""Loads a pretrained model.

        Args:
            detector_weights: The weights to load for the detector.
                One of 'L-upright' (original paper, https://arxiv.org/abs/2308.08479),
                'L-C4', 'L-SO2' (from steerers, better for rotations, https://arxiv.org/abs/2312.02152),
                'L-C4-v2' (from dedode v2, better at rotations, less clustering, https://arxiv.org/abs/2404.08928)
                Default is 'L-C4-v2', but perhaps it should be 'L-C4-v2'?
            descriptor_weights: The weights to load for the descriptor.
                One of 'B-upright','G-upright' (original paper, https://arxiv.org/abs/2308.08479),
                'B-C4', 'B-SO2', 'G-C4', 'G-SO2' (from steerers, better for rotations, https://arxiv.org/abs/2312.02152).
                Default is 'G-upright'.
            amp_dtype: the dtype to use for the model. One of torch.float16 or torch.float32.
            Default is torch.float16, suitable for CUDA. Use torch.float32 for CPU or MPS

        Returns:
            The pretrained model.
        """
        model: DeDoDe = cls(
            detector_model=detector_weights[0],  # type: ignore[arg-type]
            descriptor_model=descriptor_weights[0],  # type: ignore[arg-type]
            amp_dtype=amp_dtype,
        )
        model.detector.load_state_dict(
            torch.hub.load_state_dict_from_url(urls["detector"][detector_weights], map_location=torch.device("cpu"))
        )
        model.descriptor.load_state_dict(
            torch.hub.load_state_dict_from_url(urls["descriptor"][descriptor_weights], map_location=torch.device("cpu"))
        )
        model.eval()
        return model
