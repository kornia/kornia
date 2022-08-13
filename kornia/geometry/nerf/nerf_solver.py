from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.data_utils import Images, ImageTensors, RayDataset, instantiate_ray_dataloader
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

        self._opt_nerf: optim.Optimizer = None

    def init_training(
        self,
        cameras: PinholeCamera,
        min_depth: float,
        max_depth: float,
        imgs: Images,
        num_img_rays: Optional[torch.tensor],
        batch_size: int,
        num_ray_points: int,
        lr: float = 1.0e-3,
    ):
        # self._check_camera_image_consistency(cameras, imgs)

        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth

        self._imgs = imgs
        self._num_img_rays = num_img_rays

        self._batch_size = batch_size

        self._nerf_model = NerfModel(num_ray_points)
        self._opt_nerf = optim.Adam(self._nerf_model.parameters(), lr=lr)

    @property
    def nerf_model(self) -> nn.Module:
        return self._nerf_model

    # FIXME: Remove this function - consistency is checked in
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

    def _train_one_epoch(self) -> float:
        ray_dataset = RayDataset(self._cameras, self._min_depth, self._max_depth)
        ray_dataset.init_ray_dataset(self._num_img_rays)
        ray_dataset.init_images_for_training(self._imgs)  # FIXME: Do we need to load the same images on each Epoch?
        ray_data_loader = instantiate_ray_dataloader(ray_dataset, self._batch_size, shufle=True)
        total_loss = 0.0
        for i_batch, (origins, directions, rgbs) in enumerate(ray_data_loader):
            rgbs_model = self._nerf_model(origins, directions)
            loss = F.mse_loss(rgbs_model, rgbs)

            total_loss = total_loss + loss.item()

            self._opt_nerf.zero_grad()
            loss.backward()
            self._opt_nerf.step()
        return total_loss / i_batch

    def run(self, num_epochs=1):
        for i_epoch in range(num_epochs):
            epoch_loss = self._train_one_epoch()

            print(f'Epoch: {i_epoch}: epoch_loss = {epoch_loss}')

    def render_views(self, cameras: PinholeCamera) -> ImageTensors:
        ray_dataset = RayDataset(cameras, self._min_depth, self._max_depth)
        ray_dataset.init_ray_dataset()
        idx0 = 0
        imgs: ImageTensors = []
        for height, width in zip(cameras.height.int().tolist(), cameras.width.int().tolist()):
            idxs = list(range(idx0, idx0 + height * width))
            idx0 = idx0 + height * width
            origins, directions, _ = ray_dataset[idxs]
            rgbs_model = self._nerf_model(origins, directions)
            img = torch.tensor(
                torch.permute(rgbs_model.reshape(height, width, -1), (2, 0, 1)) * 255.0, dtype=torch.uint8
            )
            imgs.append(img)
        return imgs
