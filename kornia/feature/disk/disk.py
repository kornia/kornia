from __future__ import annotations

from typing import Optional

import torch

from kornia.core import Module, Tensor

from ._unets import Unet
from .detector import heatmap_to_keypoints
from .structs import DISKFeatures


class DISK(Module):
    r"""Module which detects and described local features in an image using the DISK method. See
    :cite:`tyszkiewicz2020disk` for details.

    .. image:: _static/img/disk_outdoor_depth.jpg

    Args:
        desc_dim: The dimension of the descriptor.
        unet: The U-Net to use. If None, a default U-Net is used. Kornia doesn't provide the training code for DISK
              so this is only useful when using a custom checkpoint trained using the code released with the paper.
              The unet should take as input a tensor of shape :math:`(B, C, H, W)` and output a tensor of shape
              :math:`(B, \mathrm{desc\_dim} + 1, H, W)`.

    Example:
        >>> disk = DISK.from_pretrained('depth')
        >>> images = torch.rand(1, 3, 256, 256)
        >>> features = disk(images)
    """

    def __init__(self, desc_dim: int = 128, unet: None | Module = None) -> None:
        super().__init__()

        self.desc_dim = desc_dim

        if unet is None:
            unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1])
        self.unet = unet

    def heatmap_and_dense_descriptors(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the heatmap and the dense descriptors.

        .. image:: _static/img/DISK.png

        Args:
            images: The image to detect features in. Shape :math:`(B, 3, H, W)`.

        Returns:
            A tuple of dense detection scores and descriptors.
            Shapes are :math:`(B, 1, H, W)` and :math:`(B, D, H, W)`, where
            :math:`D` is the descriptor dimension.
        """
        unet_output = self.unet(images)

        if unet_output.shape[1] != self.desc_dim + 1:
            raise ValueError(
                f"U-Net output has {unet_output.shape[1]} channels, but expected self.desc_dim={self.desc_dim} + 1."
            )

        descriptors = unet_output[:, : self.desc_dim]
        heatmaps = unet_output[:, self.desc_dim :]

        return heatmaps, descriptors

    def forward(
        self,
        images: Tensor,
        n: Optional[int] = None,
        window_size: int = 5,
        score_threshold: float = 0.0,
        pad_if_not_divisible: bool = False,
    ) -> list[DISKFeatures]:
        """Detects features in an image, returning keypoint locations, descriptors and detection scores.

        Args:
            images: The image to detect features in. Shape :math:`(B, 3, H, W)`.
            n: The maximum number of keypoints to detect. If None, all keypoints are returned.
            window_size: The size of the non-maxima suppression window used to filter detections.
            score_threshold: The minimum score a detection must have to be returned.
                             See :py:class:`DISKFeatures` for details.
            pad_if_not_divisible: if True, the non-16 divisible input is zero-padded to the closest 16-multiply

        Returns:
            A list of length :math:`B` containing the detected features.
        """
        B = images.shape[0]
        if pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descriptors = self.heatmap_and_dense_descriptors(images)
        if pad_if_not_divisible:
            heatmaps = heatmaps[..., :h, :w]
            descriptors = descriptors[..., :h, :w]

        keypoints = heatmap_to_keypoints(heatmaps, n=n, window_size=window_size, score_threshold=score_threshold)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return features

    @classmethod
    def from_pretrained(cls, checkpoint: str = "depth", device: torch.device = torch.device("cpu")) -> DISK:
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
            "depth": "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth",
            "epipolar": "https://raw.githubusercontent.com/cvlab-epfl/disk/master/epipolar-save.pth",
        }

        if checkpoint not in urls:
            raise ValueError(f"Unknown pretrained model: {checkpoint}")

        pretrained_dict = torch.hub.load_state_dict_from_url(urls[checkpoint], map_location=device)

        model: DISK = cls().to(device)
        model.load_state_dict(pretrained_dict["extractor"])
        model.eval()
        return model
