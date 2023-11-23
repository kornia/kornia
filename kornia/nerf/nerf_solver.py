from __future__ import annotations

import logging
from datetime import datetime
from typing import cast

import torch
import torch.nn.functional as F
from torch import optim

from kornia.core import Module, Tensor, tensor
from kornia.core.check import KORNIA_CHECK
from kornia.geometry.camera import PinholeCamera
from kornia.metrics import psnr
from kornia.nerf.core import Images
from kornia.nerf.data_utils import RayDataset, instantiate_ray_dataloader
from kornia.nerf.nerf_model import NerfModel
from kornia.utils import deprecated

logger = logging.getLogger(__name__)


class NerfSolver:
    r"""NeRF solver class.

    Args:
        device: device for class tensors: Union[str, Device]
        dtype: type for all floating point calculations: torch.dtype
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        # TODO: add support for the new CameraModel class
        # cameras used for training
        self._cameras: PinholeCamera | None = None

        # rays depth range
        self._min_depth: float = 0.0
        self._max_depth: float = 0.0

        # whether to convert ray parameters to normalized device coordinates
        self._ndc: bool = True

        # images used for training
        self._imgs: Images | None = None

        # number of rays to randomly cast from each camera
        self._num_img_rays: Tensor | int | None = None

        # number of rays to sample in a batch
        self._batch_size: int = 0

        # number of points to sample along rays
        self._num_ray_points: int = 0

        # the model and optimizer
        self._nerf_model: Module | None = None
        self._nerf_optimizer: optim.Optimizer | None = None

        self._device = device
        self._dtype = dtype

    def setup_solver(
        self,
        cameras: PinholeCamera,
        min_depth: float,
        max_depth: float,
        ndc: bool,
        imgs: Images,
        num_img_rays: Tensor | int,
        batch_size: int,
        num_ray_points: int,
        irregular_ray_sampling: bool = True,
        log_space_encoding: bool = True,
        lr: float = 1.0e-3,
    ) -> None:
        """Initializes training settings and model.

        Args:
            cameras: Scene cameras in the order of input images.
            min_depth: sampled rays minimal depth from cameras.
            max_depth: sampled rays maximal depth from cameras.
            ndc: convert ray parameters to normalized device coordinates.
            imgs: Scene 2D images (one for each camera).
            num_img_rays: Number of rays to randomly cast from each camera: math: `(B)`.
            batch_size: Number of rays to sample in a batch.
            num_ray_points: Number of points to sample along rays.
            irregular_ray_sampling: Whether to sample ray points irregularly.
            log_space: Whether frequency sampling should be log spaced.
            lr: Learning rate.
        """
        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ndc = ndc

        self._imgs = imgs

        KORNIA_CHECK(
            isinstance(batch_size, int) and batch_size > 0,
            "batch_size must be a positive integer",
        )

        KORNIA_CHECK(
            isinstance(num_ray_points, int) and num_ray_points > 0,
            "num_ray_points must be a positive integer",
        )

        KORNIA_CHECK(num_img_rays is not None, "num_img_rays must be specified")

        if isinstance(num_img_rays, int):
            self._num_img_rays = tensor([num_img_rays] * cameras.batch_size)
        elif torch.is_tensor(num_img_rays):
            self._num_img_rays = num_img_rays
        else:
            raise TypeError("num_img_rays can be either an int or a Tensor")

        self._batch_size = batch_size

        self._nerf_model = NerfModel(
            num_ray_points, irregular_ray_sampling=irregular_ray_sampling, log_space_encoding=log_space_encoding
        )
        self._nerf_model.to(device=self._device, dtype=self._dtype)

        self._nerf_optimizer = optim.Adam(self._nerf_model.parameters(), lr=lr)

    @deprecated(replace_with="setup_solver", version="0.7.0")
    def init_training(
        self,
        cameras: PinholeCamera,
        min_depth: float,
        max_depth: float,
        ndc: bool,
        imgs: Images,
        num_img_rays: Tensor | int,
        batch_size: int,
        num_ray_points: int,
        irregular_ray_sampling: bool = True,
        log_space_encoding: bool = True,
        lr: float = 1.0e-3,
    ) -> None:
        self.setup_solver(
            cameras,
            min_depth,
            max_depth,
            ndc,
            imgs,
            num_img_rays,
            batch_size,
            num_ray_points,
            irregular_ray_sampling,
            log_space_encoding,
            lr,
        )

    @property
    def nerf_model(self) -> Module | None:
        """Returns the NeRF model."""
        return self._nerf_model

    def _train_one_epoch(self) -> float:
        r"""Trains the NeRF model one epoch.

        1) A dataset of rays is initialized, and sent over to a data loader.
        2) The data loader sample a batch of rays randomly, and runs them through the NeRF model,
        to predict ray associated rgb model values.
        3) The model rgb is compared with the image pixel rgb, and the loss between the two is back
        propagated to update the model weights.

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
            Average psnr over all epoch rays.
        """
        KORNIA_CHECK(self._nerf_model is not None, "The model should be a NeRF model.")
        KORNIA_CHECK(self._nerf_optimizer is not None, "The optimizer should be an Adam optimizer.")
        KORNIA_CHECK(self._cameras is not None, "The camera should be a PinholeCamera.")
        KORNIA_CHECK(self._imgs is not None, "The images should be a list of tensors.")
        KORNIA_CHECK(self._num_img_rays is not None, "The number of images of Ray should be a tensor.")

        # TODO: refactor and so that the constructor receives the correct types
        cameras: PinholeCamera = cast(PinholeCamera, self._cameras)
        num_img_rays: Tensor = cast(Tensor, self._num_img_rays)
        images = cast(Images, self._imgs)
        nerf_model: NerfModel = cast(NerfModel, self._nerf_model)
        nerf_optimizer: optim.Optimizer = cast(optim.Optimizer, self._nerf_optimizer)

        # create the dataset and data loader
        ray_dataset = RayDataset(
            cameras, self._min_depth, self._max_depth, self._ndc, device=self._device, dtype=self._dtype
        )
        ray_dataset.init_ray_dataset(num_img_rays)
        ray_dataset.init_images_for_training(images)  # FIXME: Do we need to load the same images on each Epoch?

        # data loader
        ray_data_loader = instantiate_ray_dataloader(ray_dataset, self._batch_size, shuffle=True)

        total_psnr: Tensor = torch.tensor(0.0, device=self._device, dtype=self._dtype)

        i_batch: float = 0

        for origins, directions, rgbs in ray_data_loader:
            rgbs_model = nerf_model(origins, directions)
            loss = F.mse_loss(rgbs_model, rgbs)

            total_psnr += psnr(rgbs_model, rgbs, 1.0)

            nerf_optimizer.zero_grad()
            loss.backward()
            nerf_optimizer.step()

            i_batch += 1

        return float(total_psnr / (i_batch + 1))

    def run(self, num_epochs: int = 1) -> None:
        r"""Runs training epochs.

        Args:
            num_epochs: number of epochs to run. Default: 1.
        """
        for i_epoch in range(num_epochs):
            # train one epoch
            epoch_psnr: float = self._train_one_epoch()

            if i_epoch % 10 == 0:
                current_time = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
                logger.info("Epoch: %d: epoch_psnr = %f; time: %s", i_epoch, epoch_psnr, current_time)
