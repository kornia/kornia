from typing import Optional

from torch import Tensor
from torch.utils.data import Dataset


class CachedDataSet(Dataset):
    """
    Args:
        caching: caching methods like in-memroy, mldb, redis, spark, etc.
        resize_to: if to save the resized images.
        backend: torch, cv2, rust etc.
        device: save the image directly to cpu or gpu.
        dtype: blabalba
    """

    def __init__(
        self,
        caching: str = "in-memory",
        resize_to: Optional[tuple] = None,
        backend: str = "torch",
        device: str = None,
        dtype=None,
        transform=None
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype
        self.resize_to = resize_to
        self.backend = backend
        self.transform = transform

        if caching == "in-memory":
            self.cache = InMemoryCache()
        else:
            raise ValueError

    def __len__(self,):
        return len(self.data)

    def __readimage__(self, index):
        raise NotImplementedError

    # This shall outputs encapsulated Image, BBox, Keypoints, etc.
    def __getitem__(self, index):
        img = self.cache.get(index)
        if img is None:
            img = super().__readimage__(index)
            self.cache.set(index, img)
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageCache:

    def get(self, key: str) -> Optional[Tensor]:
        raise NotImplementedError

    def set(self, key: str, value: Tensor) -> None:
        raise NotImplementedError


class InMemoryCache(ImageCache):

    def __init__(self) -> None:
        super().__init__()
        self.data = {}

    def get(self, key: str) -> Tensor:
        return self.data[key]

    def set(self, key: str, value: Tensor) -> None:
        self.data[key] = value
