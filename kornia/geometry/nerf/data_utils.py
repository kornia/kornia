from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.io import read_image

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.rays import RandomRaySampler, RaySampler, UniformRaySampler
from kornia.geometry.nerf.types import Device

ImagePaths = List[str]
ImageTensors = List[torch.Tensor]
Images = Union[ImagePaths, ImageTensors]

RayGroup = Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]


class RayDataset(Dataset):  # FIXME: Add device
    def __init__(self, cameras: PinholeCamera, min_depth: float, max_depth: float, device: Device) -> None:
        super().__init__()
        self._ray_sampler: Optional[RaySampler] = None
        self._imgs: Optional[List[torch.Tensor]] = None
        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._device = device

    def init_ray_dataset(self, num_img_rays: Optional[torch.Tensor] = None) -> None:
        if num_img_rays is None:
            self._init_uniform_ray_dataset()
        else:
            self._init_random_ray_dataset(num_img_rays)

    def init_images_for_training(self, imgs: Images) -> None:
        self._check_image_type_consistency(imgs)
        if isinstance(imgs[0], str):  # Load images from disk
            self._imgs = self._load_images(imgs)
        else:
            self._imgs = imgs  # Take images provided on input
        self._check_dimensions(self._imgs)

        # Move images to defined device
        self._imgs = [img.to(self._device) for img in self._imgs]

    def _init_random_ray_dataset(self, num_img_rays: torch.Tensor):
        self._ray_sampler = RandomRaySampler(self._min_depth, self._max_depth, device=self._device)
        self._ray_sampler.calc_ray_params(self._cameras, num_img_rays)

    def _init_uniform_ray_dataset(self):
        self._ray_sampler = UniformRaySampler(self._min_depth, self._max_depth, device=self._device)
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
        return len(self._ray_sampler)

    def __getitem__(self, idxs: Union[int, List[int]]) -> RayGroup:
        origins = self._ray_sampler.origins[idxs]
        directions = self._ray_sampler.directions[idxs]
        if self._imgs is None:
            return origins, directions, None
        camerd_ids = self._ray_sampler.camera_ids[idxs]
        points_2d = self._ray_sampler.points_2d[idxs]
        rgbs = None
        imgs_for_ids = [self._imgs[i] for i in camerd_ids]
        rgbs = torch.stack(
            [img[:, point2d[1].item(), point2d[0].item()] for img, point2d in zip(imgs_for_ids, points_2d)]
        )
        rgbs = rgbs.float() / 255.0
        return origins, directions, rgbs


def instantiate_ray_dataloader(dataset: RayDataset, batch_size: int = 1, shufle: bool = True):
    def collate_rays(items: List[RayGroup]) -> RayGroup:
        return items[0]

    return DataLoader(
        dataset,
        sampler=BatchSampler(
            RandomSampler(dataset) if shufle else SequentialSampler(dataset), batch_size, drop_last=False
        ),
        collate_fn=collate_rays,
    )
