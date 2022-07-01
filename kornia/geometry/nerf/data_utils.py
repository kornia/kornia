from typing import Any, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.rays import RandomRaySampler, RaySampler, UniformRaySampler


class RayDataset(Dataset):  # FIXME: Add device
    def __init__(self, cameras: PinholeCamera, img_paths: List[str], min_depth: float, max_depth: float) -> None:
        super().__init__()
        self._ray_sampler: Optional[RaySampler] = None
        self._imgs: List[torch.Tensor] = []
        self._cameras = cameras
        self._img_paths = img_paths
        self._min_depth = min_depth
        self._max_depth = max_depth

    def init_random_ray_dataset(self, num_rays: torch.Tensor):
        self._ray_sampler = RandomRaySampler(self._min_depth, self._max_depth)
        self._ray_sampler.calc_ray_params(self._cameras, num_rays)
        self._imgs = self._load_images(self._img_paths)  # FIXME: Add consistency check images and cameras

    def init_uniform_ray_dataset(self):
        self._ray_sampler = UniformRaySampler(self._min_depth, self._max_depth)
        self._ray_sampler.calc_ray_params(self._cameras)
        self._imgs = self._load_images(self._img_paths)  # FIXME: Add consistency check images and cameras

    @staticmethod
    def _load_images(img_paths: List[str]) -> List[torch.Tensor]:
        imgs: List[torch.Tensor] = []
        for img_path in img_paths:
            imgs.append(read_image(img_path))
        return imgs

    def __len__(self):
        pass

    def __getitem__(self, idx: int) -> Any:
        return super().__getitem__(idx)
