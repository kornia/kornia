from typing import Optional

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.data_utils import Images, RayDataset, instantiate_ray_dataloader
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


class NerfSolver:
    def __init__(self) -> None:
        self._cameras: Optional[PinholeCamera] = None
        self._min_depth: float = 0.0
        self._max_depth: float = 0.0

        self._imgs: Optional[Images] = None
        self._num_img_rays: Optional[int] = None

        self._batch_size: int = 0

        self._nerf_model: Optional[nn.Module] = None
        self._num_ray_points: int = 0

        self._nerf_optimizaer: Optional[optim.Optimizer] = None

    def init_training(
        self,
        cameras: PinholeCamera,
        min_depth: float,
        max_depth: float,
        imgs: Images,
        batch_size: int,
        num_ray_points: int,
    ):
        self._check_camera_image_consistency(cameras, imgs)

        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth

        self._imgs = imgs
        self._num_img_rays = None

        self._batch_size = batch_size

        self._nerf_model = NerfModel(num_ray_points)

    @staticmethod
    def _check_camera_image_consistency(cameras: PinholeCamera, imgs: Images):
        if cameras is None:
            raise ValueError('Invalid camera object')
        if imgs is None:
            raise ValueError('Invalid image list object')
        if cameras.batch_size != len(imgs):
            raise ValueError('Number of cameras must match number of input images')
        if not all(img.shape[0] == 3 for img in imgs):
            raise ValueError('All images must have three RGB channels')
        if not all(height == img.shape[1] for height, img in zip(cameras.height.tolist(), imgs)):
            raise ValueError('All image heights must match camera heights')
        if not all(width == img.shape[2] for width, img in zip(cameras.width.tolist(), imgs)):
            raise ValueError('All image widths must match camera widths')

    def train_one_epoch(self):
        ray_dataset = RayDataset(self._cameras, self._min_depth, self._max_depth)
        ray_dataset.init_ray_dataset(self._imgs, self._num_img_rays)
        ray_data_loader = instantiate_ray_dataloader(ray_dataset, self._batch_size, shufle=True)
        for origins, directions, rgbs in ray_data_loader:
            rgbs_model = self._nerf_model(origins, directions)
            loss = F.mse_loss(rgbs_model, rgbs)

            loss.backward()

            # opt_nerf.step()
            # opt_nerf.zero_grad()
