from __future__ import annotations

import torch.nn
from torch.utils.data import Dataset

from kornia.augmentation import (
    ColorJiggle,
    ColorJitter,
    RandomAffine,
    RandomAffine3D,
    RandomCrop,
    RandomCrop3D,
    RandomCutMixV2,
    RandomErasing,
    RandomGaussianBlur,
    RandomJigsaw,
    RandomMixUpV2,
    RandomMosaic,
    RandomMotionBlur,
    RandomMotionBlur3D,
    RandomPerspective,
    RandomPerspective3D,
    RandomPlanckianJitter,
    RandomPosterize,
    RandomRain,
    RandomRotation3D,
    RandomShear,
    RandomTranslate,
    Resize,
)


class DummyMPDataset(Dataset):
    def __init__(self, context: str):
        super().__init__()
        # we add all transforms that could potentially fail in
        # multiprocessing with a spawn context below, that is all the
        # transforms that define a RNG
        transforms = [
            RandomTranslate(),
            RandomShear(0.1),
            RandomPosterize(),
            RandomErasing(),
            RandomMotionBlur(kernel_size=3, angle=(0, 360), direction=(-1, 1)),
            RandomGaussianBlur(3, (0.1, 2.0)),
            RandomPerspective(),
            ColorJitter(),
            ColorJiggle(),
            RandomJigsaw(),
            RandomAffine(degrees=15),
            RandomMotionBlur3D(kernel_size=3, angle=(0, 360), direction=(-1, 1)),
            RandomPerspective3D(),
            RandomAffine3D(degrees=15),
            RandomRotation3D(degrees=15),
        ]

        if context != "fork":
            # random planckian jitter auto selects a GPU. But it is not possible
            # to init a CUDA context in a forked process.
            # So we skip it in this case.
            transforms.append(RandomPlanckianJitter())

        self._transform = torch.nn.Sequential()

        self._resize = Resize((10, 10))
        self._mosaic = RandomMosaic((2, 2))
        self._crop = RandomCrop((5, 5))
        self._crop3d = RandomCrop3D((5, 5, 5))
        self._mixup = RandomMixUpV2()
        self._cutmix = RandomCutMixV2()
        self._rain = RandomRain(p=1, drop_height=(1, 2), drop_width=(1, 2), number_of_drops=(1, 1))

    def __len__(self):
        return 10

    def __getitem__(self, _):
        mosaic = self._mosaic(torch.rand(1, 3, 64, 64))
        rain = self._rain(torch.rand(1, 1, 5, 5))
        rain = self._resize(rain)
        cropped = self._crop(torch.rand(3, 3, 64, 64))
        cropped3d = self._crop3d(torch.rand(3, 64, 64, 64))
        mixed = self._mixup(torch.rand(3, 3, 64, 64), torch.rand(3, 3, 64, 64))
        mixed = self._cutmix(torch.rand(3, 3, 64, 64), mixed)

        return (self._transform(mixed), cropped, cropped3d, mixed, mosaic, rain)
