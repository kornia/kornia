from typing import Any, List, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from torchvision.io import read_image

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.rays import RandomRaySampler, RaySampler, UniformRaySampler

ImagePaths = List[str]
ImageTensors = List[torch.Tensor]
Images = Union[ImagePaths, ImageTensors]


class DatasetItem:
    def __init__(self, origin: torch.Tensor, direction: torch.Tensor, rgb: torch.Tensor) -> None:
        self._origin = origin
        self._direction = direction
        self._rgb = rgb


class RayDataset(Dataset):  # FIXME: Add device
    def __init__(self, cameras: PinholeCamera, min_depth: float, max_depth: float) -> None:
        super().__init__()
        self._ray_sampler: Optional[RaySampler] = None
        self._imgs: List[torch.Tensor] = []
        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth

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
        if not all(isinstance(img, str) for img in imgs) and not all(isinstance(img, torch.Tensor) for img in imgs):
            raise ValueError('The list of input images can only be all paths or tensors')

    def _check_dimensions(self, imgs: ImageTensors):
        if len(imgs) != self._cameras.batch_size:
            raise ValueError(
                f'Number of images {len(imgs)} does not match number of cameras {self._cameras.batch_size}'
            )
        if not all(img.shape[0] == 3 for img in imgs):
            raise ValueError('Not all input images have 3 channels')
        for i, (img, height, width) in enumerate(zip(imgs, self._cameras.height, self._cameras.width)):
            if img.shape[1:] != (height, width):
                raise ValueError(
                    f'Image index {i} dimensions {(img.shape[1], img.shape[2])} are inconsistent with equivalent '
                    f'camera dimensions {(height.item(), width.item())}'
                )

    @staticmethod
    def _load_images(img_paths: List[str]) -> List[torch.Tensor]:
        imgs: List[torch.Tensor] = []
        for img_path in img_paths:
            imgs.append(read_image(img_path))
        return imgs

    def __len__(self):
        return self._ray_sampler.origins.shape[0]

    def __getitem__(self, idx: Union[int, List[int]]) -> Any:
        origin = self._ray_sampler.origins[idx]
        direction = self._ray_sampler.directions[idx]
        camerd_id = self._ray_sampler.camera_ids[idx]
        point_2d = self._ray_sampler.points_2d[idx]
        x = point_2d[0].item()
        y = point_2d[1].item()
        rgb = self._imgs[camerd_id][:, y, x]
        return DatasetItem(origin, direction, rgb)


class RayDataloader(DataLoader):
    def __init__(self, dataset: RayDataset, batch_size: int = 1):
        super().__init__(
            dataset,
            sampler=BatchSampler(RandomSampler(dataset), batch_size, drop_last=False),
            collate_fn=self._collate_rays,
        )

    @staticmethod
    def _collate_rays(dataset_items: List[DatasetItem]) -> None:
        origins = [dataset_item._origin for dataset_item in dataset_items]
        directions = [dataset_item._direction for dataset_item in dataset_items]
        rgbs = [dataset_item._rgb for dataset_item in dataset_items]
        return torch.stack(origins), torch.stack(directions), torch.stack(rgbs)
