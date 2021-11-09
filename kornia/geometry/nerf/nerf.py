import unittest
from typing import Tuple

import torch
import torch.nn as nn

from kornia.geometry.nerf.from_nerfmm import train_model


class CameraCalibration:
    def __init__(self, nerf_model: torch.nn.Module, n_epoch: int = 1):
        self._nerf_model = nerf_model
        self._images: torch.Tensor = None
        self._nerf_model: nn.Module = None
        self._focal_net: nn.Module = None
        self._pose_param_net: nn.Module = None

        self._n_epoch = n_epoch

    def add_images(self, images: torch.Tensor):
        self._images = images

    def run(self):
        self._nerf_model, self._focal_net, self._pose_param_net = \
            train_model(self._images, n_epoch = self._n_epoch)

    def model(self) -> nn.Module:
        return self._nerf_model

    def focals(self) -> Tuple[float, float]:
        fxfy = self._focal_net()
        return fxfy[0], fxfy[1]

    def pose(self, camera_id: int):
        return self._pose_param_net(camera_id)

    def n_epoch(self) -> int:
        return self._n_epoch

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
