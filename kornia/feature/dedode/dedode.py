from typing import Optional

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor

from .dedode_models import DeDoDeDescriptor, DeDoDeDetector, get_descriptor, get_detector
from .utils import sample_keypoints


class DeDoDe(Module):
    r"""Module which detects and/or describes local features in an image using the DeDode method. See
    :cite:`edstedt2024dedode` for details.

    TODO: implement steerers and mnn matchers
    NOTE: DeDode takes ImageNet normalized images as input (not in range [0, 1]).
    Example:
        >>> dedode = DeDoDe.from_pretrained(weights = "default")
        >>> images = torch.randn(1, 3, 256, 256)
        >>> detections = dedode.detect(images)
        >>> descriptions = dedode.describe(images, detections = detections)
        >>> detections, features = dedode(images) # alternatively do both
    """

    def __init__(self, detector_model="L", descriptor_model="G") -> None:
        super().__init__()
        self.detector: DeDoDeDetector = get_detector[detector_model]()
        self.descriptor: DeDoDeDescriptor = get_descriptor[descriptor_model]()

    def forward(
        self,
        images: Tensor,
        n: Optional[int] = 10_000,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            A list of length :math:`B` containing the detected features.
        """
        keypoints, scores = self.detect(images, n=n)
        descriptions = self.describe(images, keypoints)
        return keypoints, scores, descriptions

    @torch.inference_mode()
    def detect(self, images, n: Optional[int] = 10_000):
        self.train(False)
        B, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"DeDoDe expects RGB, got {C=}")
        logits = self.detector.forward(images)
        scoremap = logits.reshape(B, H * W).softmax(dim=-1).reshape(B, H, W)
        keypoints, confidence = sample_keypoints(scoremap, num_samples=n)
        return keypoints, confidence

    @torch.inference_mode()
    def describe(self, images, keypoints=None):
        self.train(False)
        B, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"DeDoDe expects RGB, got {C=}")
        descriptions = self.descriptor.forward(images)
        if keypoints is not None:
            described_keypoints = F.grid_sample(
                descriptions.float(), keypoints[:, None], mode="bilinear", align_corners=False
            )[:, :, 0].mT
            return described_keypoints
        else:
            return descriptions

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: str = "L-upright",
        descriptor_weights: str = "G-upright",
        device: torch.device = torch.device("cpu"),
    ):
        r"""Loads a pretrained model.

        Depth model was trained using depth map supervision and is slightly more precise but biased to detect keypoints
        only where SfM depth is available. Epipolar model was trained using epipolar geometry supervision and
        is less precise but detects keypoints everywhere where they are matchable. The difference is especially
        pronounced on thin structures and on edges of objects.

        Args:
            checkpoint: The checkpoint to load. One of 'depth' or 'epipolar'.
            device: The device to load the model to.

        Returns:
            The pretrained model.
        """
        urls = {
            "detector": {
                "L-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth",
                "L-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_C4.pth",
                "L-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/dedode_detector_SO2.pth",
            },
            "descriptor": {
                "B-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth",
                "B-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_C4_Perm_descriptor_setting_C.pth",
                "B-SO2": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/B_SO2_Spread_descriptor_setting_C.pth",
                "G-upright": "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth",
                "G-C4": "https://github.com/georg-bn/rotation-steerers/releases/download/release-2/G_C4_Perm_descriptor_setting_C.pth",
            },
        }

        model: DeDoDe = cls(detector_model=detector_weights[0], descriptor_model=descriptor_weights[0]).to(device)
        model.detector.load_state_dict(urls["detector"][detector_weights])
        model.descriptor.load_state_dict(urls["descriptor"][descriptor_weights])
        model.eval()
        return model
