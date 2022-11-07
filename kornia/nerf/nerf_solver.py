from datetime import datetime
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from kornia.core import Module, Tensor, tensor
from kornia.geometry.camera import PinholeCamera
from kornia.metrics import psnr
from kornia.nerf.core import Images, ImageTensors
from kornia.nerf.data_utils import RayDataset, instantiate_ray_dataloader
from kornia.nerf.nerf_model import NerfModel
from kornia.utils._compat import torch_inference_mode


class NerfSolver:
    r"""NeRF solver class.

    Args:
        device: device for class tensors: Union[str, torch.device]
        dtype: type for all floating point calculations: torch.dtype
    """
    _nerf_model: Module
    _opt_nerf: optim.Optimizer

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self._cameras: Optional[PinholeCamera] = None
        self._min_depth: float = 0.0
        self._max_depth: float = 0.0
        self._ndc: bool = True

        self._imgs: Optional[Images] = None
        self._num_img_rays: Optional[Union[Tensor, int]] = None

        self._batch_size: int = 0

        self._num_ray_points: int = 0

        self._nerf_optimizaer: Optional[optim.Optimizer] = None

        self._device = device
        self._dtype = dtype

    def init_training(
        self,
        cameras: PinholeCamera,
        min_depth: float,
        max_depth: float,
        ndc: bool,
        imgs: Images,
        num_img_rays: Optional[Union[Tensor, int]],
        batch_size: int,
        num_ray_points: int,
        irregular_ray_sampling: bool = True,
        log_space_encoding=True,
        num_hidden=256,
        lr: float = 1.0e-3,
    ) -> None:
        r"""Initializes training.

        Args:
            cameras: Scene cameras in the order of input images: PinholeCamera
            min_depth: sampled rays minimal depth from cameras: float
            max_depth: sampled rays maximal depth from cameras: float
            ndc: convert ray parameters to normalized device coordinates: bool
            imgs: Scene 2D images (one for each camera): Images
            num_img_rays: Number of rays to randomly cast from each camera: math: `(B)`
            batch_size: Number of rays to sample in a batch: int
            num_ray_points: Number of points to sample along rays: int
            irregular_ray_sampling: Whether to sample ray points irregularly: bool
            log_space_encoding: Whether frequency sampling should be log spaced: bool
            num_hidden: Layer hidden dimensions: int
            lr: Learning rate: float
        """
        # self._check_camera_image_consistency(cameras, imgs)   # FIXME: Check if required

        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ndc = ndc

        self._imgs = imgs

        if num_img_rays is None:
            self._num_img_rays = None
        elif isinstance(num_img_rays, int):
            self._num_img_rays = tensor([num_img_rays] * cameras.batch_size)
        elif torch.is_tensor(num_img_rays):
            self._num_img_rays = num_img_rays
        else:
            raise TypeError('num_img_rays can be either an int or a Tensor')

        self._batch_size = batch_size

        self._nerf_model = NerfModel(
            num_ray_points,
            irregular_ray_sampling=irregular_ray_sampling,
            log_space_encoding=log_space_encoding,
            num_hidden=num_hidden,
        )
        self._nerf_model.to(device=self._device, dtype=self._dtype)
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
        r"""Trains one epoch. A dataset of rays is initialized, and sent over to a data loader. The data loader
        sample a batch of rays randomly, and runs them through the NeRF model, to predict ray associated rgb model
        values. The model rgb is compared with the image pixel rgb, and the loss between the two is back propagated
        to update the model weights.

        Implemented steps:
        - Create an object of class RayDataset
        - Initialize ray dataset with group of images on disk, and number of rays to randomly sample
        - Initialize a data loader with batch size info
        - Iterate over data loader
        -- Reset optimizer
        -- Run ray batch through Nerf model
        -- Find loss
        -- Back propagate loss
        -- Optimizer step

        Returns:
            Average psnr over all epoch rays
        """
        ray_dataset = RayDataset(
            self._cameras, self._min_depth, self._max_depth, self._ndc, device=self._device, dtype=self._dtype
        )
        ray_dataset.init_ray_dataset(self._num_img_rays)
        ray_dataset.init_images_for_training(self._imgs)  # FIXME: Do we need to load the same images on each Epoch?
        ray_data_loader = instantiate_ray_dataloader(ray_dataset, self._batch_size, shuffle=True)
        total_psnr = tensor(0.0, device=self._device, dtype=self._dtype)
        for i_batch, (origins, directions, rgbs) in enumerate(ray_data_loader):
            rgbs_model = self._nerf_model(origins, directions)
            loss = F.mse_loss(rgbs_model, rgbs)

            total_psnr = psnr(rgbs_model, rgbs, 1.0) + total_psnr

            self._opt_nerf.zero_grad()
            loss.backward()
            self._opt_nerf.step()
        return float(total_psnr / (i_batch + 1))

    def run(self, num_epochs: int = 1) -> None:
        r"""Runs training epochs.

        Args:
            num_epochs: Number of epochs to run: int
        """
        for i_epoch in range(num_epochs):
            epoch_psnr = self._train_one_epoch()

            if i_epoch % 10 == 0:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f'Epoch: {i_epoch}: epoch_psnr = {epoch_psnr}; time: {current_time}')

    def render_views(self, cameras: PinholeCamera) -> ImageTensors:
        r"""Renders a novel synthesis view of a trained NeRF model for given cameras.

        Args:
            cameras: cameras for image renderings: PinholeCamera

        Returns:
            Rendered images: ImageTensors (List[(H, W, C)]).
        """
        ray_dataset = RayDataset(
            cameras, self._min_depth, self._max_depth, self._ndc, device=self._device, dtype=self._dtype
        )
        ray_dataset.init_ray_dataset()
        idx0 = 0
        imgs: ImageTensors = []
        batch_size = 4096  # FIXME: Consider exposing this value to the user
        for height, width in zip(cameras.height.int().tolist(), cameras.width.int().tolist()):
            bsz = batch_size if batch_size != -1 else height * width
            img = torch.zeros((height * width, 3), dtype=torch.uint8)
            idx0_camera = idx0
            for idx0 in range(idx0, idx0 + height * width, bsz):
                idxe = min(idx0 + bsz, idx0_camera + height * width)
                idxs = list(range(idx0, idxe))
                origins, directions, _ = ray_dataset[idxs]
                with torch_inference_mode():
                    rgb_model = self._nerf_model(origins, directions) * 255.0
                    img[idx0 - idx0_camera : idxe - idx0_camera] = rgb_model
            idx0 = idxe
            img = img.reshape(height, width, -1)  # (H, W, C)
            imgs.append(img)
        return imgs
