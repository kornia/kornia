from typing import Optional

import torch.optim as optim
from data_utils import Images
from torch import nn

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.data_utils import RayDataset, instantiate_ray_dataloader
from kornia.geometry.nerf.nerf_model import NerfModel

"""
General design for Nerf computation:
===================================
Training one epoch:
------------------
- Create an object of class RayDataset
- Initialize ray dataset with group of images on disk, and number of rays to randomly sample
- Initialize a data loader with batch size info
- Iterate over data loader
-- Reset optimizer
-- Run ray batch through Nerf model
-- Find loss
-- Back propagate loss
-- Optimizer step
"""


class Nerf:
    def __init__(self) -> None:
        self._cameras: Optional[PinholeCamera] = None
        self._min_depth: float = 0.0
        self._max_depth: float = 0.0

        self._imgs: Optional[Images] = None
        self._num_img_rays: int = None

        self._batch_size: int = 0

        self._nerf_model: Optional[nn.Module] = None
        self._num_ray_points: int = 0

        self._nerf_optimizaer: Optional[optim.Optimizer] = None

    def init_training(self, imgs: Images):
        self._nerf_model = NerfModel(self._num_ray_points)

    def __train_one_epoch(self):
        ray_dataset = RayDataset(self._cameras, self._min_depth, self._max_depth)
        ray_dataset.init_ray_dataset(self._imgs, self._num_img_rays)
        ray_data_loader = instantiate_ray_dataloader(ray_dataset, self._batch_size, shufle=True)
        for origins, directions, rgbs in ray_data_loader:
            densities, rgbs = self._nerf_model(origins, directions)

            print(densities, rgbs)
