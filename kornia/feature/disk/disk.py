from typing import Literal, Tuple
import torch
import numpy as np
from torch import Tensor

from kornia.utils.helpers import map_location_to_cpu

from .unets import Unet, thin_setup
from .detector import Detector

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}

class DISK(torch.nn.Module):
    def __init__(
        self,
        desc_dim=128,
        window=8,
        setup=DEFAULT_SETUP,
        kernel_size=5,
    ):
        super(DISK, self).__init__()

        self.desc_dim = desc_dim
        self.unet = Unet(
            in_features=3, size=kernel_size,
            down=[16, 32, 64, 64, 64],
            up=[64, 64, 64, desc_dim+1],
            setup=setup,
        )
        self.detector = Detector(window=window)

    def _split(self, unet_output: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        descriptors = unet_output[:, :self.desc_dim]
        heatmap     = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    def detect(
        self,
        images: Tensor,
        kind: Literal['rng', 'nms'] = 'rng',
        **kwargs
    ) -> np.ndarray:
        ''' allowed values for `kind`:
            * rng
            * nms
        '''

        B = images.shape[0]
        try:
            descriptors, heatmaps = self._split(self.unet(images))
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[kind](heatmaps, **kwargs)

        features = []
        for i in range(B):
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        return np.array(features, dtype=object)

class Disk(DISK):
    def __init__(
        self,
        pretrained: Literal['none', 'depth', 'epipolar'] = 'none',
        device: torch.device = torch.device('cpu'),
    ):
        super(Disk, self).__init__()

        if pretrained == 'depth':
            pretrained_dict = torch.hub.load_state_dict_from_url(
                'https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth',
                map_location=map_location_to_cpu,
            )

            self.load_state_dict(pretrained_dict['extractor'])
        else:
            raise ValueError()
        
        self.to(device)