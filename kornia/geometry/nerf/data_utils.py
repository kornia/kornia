from typing import Any, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.rays import RandomRaySampler, RaySampler, UniformRaySampler


class RayDataset(Dataset):  # FIXME: Add device
    def __init__(self, cameras: PinholeCamera, min_depth: float, max_depth: float) -> None:
        super().__init__()
        self._ray_sampler: Optional[RaySampler] = None
        self._imgs: List[torch.Tensor] = []
        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth

    ImagePaths = List[str]
    ImageTensors = List[torch.Tensor]
    Images = Union[ImagePaths, ImageTensors]

    def init_ray_dataset(self, imgs: Images, num_rays: Optional[torch.Tensor] = None) -> None:
        self._check_image_type_consistency(imgs)
        if num_rays is None:
            self._init_uniform_ray_dataset()
        else:
            self._init_random_ray_dataset(num_rays)
        if isinstance(imgs[0], str):
            self._imgs = self._load_images(imgs)
        else:
            self._imgs = imgs
        self._check_dimensions(self._imgs)

    def _init_random_ray_dataset(self, num_rays: torch.Tensor):
        self._ray_sampler = RandomRaySampler(self._min_depth, self._max_depth)
        self._ray_sampler.calc_ray_params(self._cameras, num_rays)

    def _init_uniform_ray_dataset(self):
        self._ray_sampler = UniformRaySampler(self._min_depth, self._max_depth)
        self._ray_sampler.calc_ray_params(self._cameras)

    def _check_image_type_consistency(self, imgs: Images):
        first_img_type = type(imgs[0])
        if not isinstance(first_img_type, str) and not isinstance(first_img_type, torch.Tensor):
            raise ValueError('Input images can be a list of either paths or tensors')
        if not all(isinstance(type(img), first_img_type) for img in imgs):
            raise ValueError('The list of input images can only be all paths or tensors')

    def _check_dimensions(self, imgs: ImageTensors):
        if len(imgs) != self._cameras.batch_size():
            raise ValueError(
                f'Number of images {len(imgs)} does not match number of cameras {self._cameras.batch_size()}'
            )
        if not all(img.shape[0] == 3 for img in imgs):
            raise ValueError('Not all input images have 3 channels')
        if not all(
            img.shape[1:] == (height, width)
            for img, height, width in zip(imgs, self._cameras.height, self._cameras.width)
        ):
            raise ValueError('Inconsistent camera and input image dimensions')

    @staticmethod
    def _load_images(img_paths: List[str]) -> List[torch.Tensor]:
        imgs: List[torch.Tensor] = []
        for img_path in img_paths:
            imgs.append(read_image(img_path))
        return imgs

    def __len__(self):
        self._ray_sampler.origins().shape[0]

    def __getitem__(self, idx: int) -> Any:  # FIXME: idx as a list for batch processing?
        return self._ray_sampler.origins()[idx], self._ray_sampler.directions()[idx]
