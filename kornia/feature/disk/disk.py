from typing import List, Optional, Tuple

import torch

from kornia.core import Tensor

from .detector import heatmap_to_keypoints
from .structs import DISKFeatures
from .unets import Unet


class DISK(torch.nn.Module):
    r"""Module which detects and described local features in an image using the DISK method. See
    :cite:`tyszkiewicz2020disk` for details.

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

    def __init__(self, desc_dim=128, unet=None):
        super().__init__()

        self.desc_dim = desc_dim

        if unet is None:
            unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1])
        self.unet = unet

    def heatmap_and_dense_descriptors(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the heatmap and the dense descriptors."""

        try:
            unet_output = self.unet(images)
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                raise ValueError(
                    f'The current implementation of U-Net requires image size to '
                    f'be divisible by 16 (got {images.shape[2:]}).'
                )
            else:
                raise

        if unet_output.shape[1] != self.desc_dim + 1:
            raise ValueError(
                f'U-Net output has {unet_output.shape[1]} channels, but expected self.desc_dim={self.desc_dim} + 1.'
            )

        descriptors = unet_output[:, : self.desc_dim]
        heatmaps = unet_output[:, self.desc_dim :]

        return heatmaps, descriptors

    def forward(
        self, images: Tensor, n: Optional[int] = None, window_size: int = 5, score_threshold: float = 0.0
    ) -> List[DISKFeatures]:
        """Detects features in an image, returning keypoint locations, descriptors and detection scores.

        Args:
            images: The image to detect features in. Shape :math:`(B, C, H, W)`.
            n: The maximum number of keypoints to detect. If None, all keypoints are returned.
            window_size: The size of the non-maxima suppression window used to filter detections.
            score_threshold: The minimum score a detection must have to be returned.
                             See :py:class:`DISKFeatures` for details.

        Returns:
            A list of length :math:`B` containing the detected features.
        """
        B = images.shape[0]
        heatmaps, descriptors = self.heatmap_and_dense_descriptors(images)
        keypoints = heatmap_to_keypoints(heatmaps, n=n, window_size=window_size, score_threshold=score_threshold)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return features

    @classmethod
    def from_pretrained(cls, checkpoint: str = 'depth', device: torch.device = torch.device('cpu')) -> 'DISK':
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
            'depth': 'https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth',
            'epipolar': 'https://raw.githubusercontent.com/cvlab-epfl/disk/master/epipolar-save.pth',
        }

        if checkpoint not in urls:
            raise ValueError(f'Unknown pretrained model: {checkpoint}')

        pretrained_dict = torch.hub.load_state_dict_from_url(urls[checkpoint], map_location=device)

        model: DISK = cls().to(device)
        model.load_state_dict(pretrained_dict['extractor'])
        model.eval()
        return model
