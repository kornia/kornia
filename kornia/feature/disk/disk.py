from typing import Literal, Sequence, Tuple

import torch
from torch import Tensor

from kornia.utils.helpers import map_location_to_cpu

from .detector import Detector
from .structs import DISKFeatures, Keypoints
from .unets import Unet, thin_setup

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}


class DISK(torch.nn.Module):
    def __init__(self, desc_dim=128, window=8, setup=DEFAULT_SETUP, kernel_size=5):
        super().__init__()

        self.desc_dim = desc_dim
        self.unet = Unet(
            in_features=3, size=kernel_size, down=[16, 32, 64, 64, 64], up=[64, 64, 64, desc_dim + 1], setup=setup
        )
        self.detector = Detector(window=window)

    def heatmap_and_dense_descriptors(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the heatmap and the dense descriptors."""

        try:
            unet_output = self.unet(images)
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = (
                    'U-Net failed because the input is of wrong shape. With '
                    'a n-step U-Net (n == 4 by default), input images have '
                    'to have height and width as multiples of 2^n (16 by '
                    'default).'
                )
                raise RuntimeError(msg) from e
            else:
                raise

        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, : self.desc_dim]
        heatmaps = unet_output[:, self.desc_dim :]

        return heatmaps, descriptors

    def detect(self, images: Tensor, algorithm: Literal['rng', 'nms'] = 'nms', **kwargs) -> Sequence[DISKFeatures]:
        """allowed values for `algorithm`:

        * rng
        * nms
        """

        B = images.shape[0]
        heatmaps, descriptors = self.heatmap_and_dense_descriptors(images)

        keypoints: Sequence[Keypoints] = {'rng': self.detector.sample, 'nms': self.detector.nms}[algorithm](
            heatmaps, **kwargs
        )

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return features

    @classmethod
    def from_pretrained(
        cls, checkpoint: Literal['depth', 'epipolar'] = 'depth', device: torch.device = torch.device('cpu')
    ) -> 'DISK':
        if checkpoint == 'depth':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth',
                map_location=map_location_to_cpu,
            )
        elif checkpoint == 'epipolar':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://raw.githubusercontent.com/cvlab-epfl/disk/master/epipolar-save.pth',
                map_location=map_location_to_cpu,
            )
        else:
            raise ValueError(f'Unknown pretrained model: {checkpoint}')

        model: DISK = cls()
        model.load_state_dict(pretrained_dict['extractor'])
        model.eval()
        model.to(device)
        return model
