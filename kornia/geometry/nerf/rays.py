from typing import Dict, List, Tuple

import torch

from kornia.geometry.camera import PinholeCamera


class RaySampler:
    class Points2DBase:
        _camera_ids: List[int] = []

    class Points2D(Points2DBase):
        _points_2d: torch.Tensor  # (N, 2)

    class Points2DAsList(Points2DBase):  # FIXME: Remove this class
        _points_2d: List[Tuple[float, float]] = []

    _points_2d_camera: Dict[int, Points2D] = {}

    def __init__(self, heights: torch.Tensor, widths: torch.Tensor) -> None:
        self._heights: torch.Tensor = heights
        self._widths: torch.Tensor = widths

    def sample_points() -> None:
        raise NotImplementedError


class RandomRaySampler(RaySampler):
    class CameraDims:
        _heights: List[int] = []
        _widths: List[int] = []
        _camera_ids: List[int] = []

    def __init__(self, heights: torch.Tensor, widths: torch.Tensor, num_rays: torch.Tensor) -> None:
        super().__init__(heights, widths)
        self._num_rays: torch.Tensor = num_rays

    def sample_points(self) -> None:
        # points_2d_camera: Dict[int, RaySampler.Points2DAsList] = {}

        cameras_dims: Dict[int, RandomRaySampler.CameraDims] = {}
        for camera_id, height, width, n in enumerate(zip(self._heights, self._widths, self._num_rays)):
            if n not in cameras_dims:
                cameras_dims[n] = RandomRaySampler.CameraDims()
            cameras_dims[n]._heights.extend([height] * n)
            cameras_dims[n]._widths.extend([width] * n)
            cameras_dims[n]._camera_ids.append(camera_id)

        for n, camera_dims in cameras_dims.items():
            dims = torch.tensor([camera_dims._heights, camera_dims._widths], dtype=torch.float32).T
            points_2d = torch.trunc(torch.rand_like(dims, dtype=torch.float32) * dims).reshape(-1, n, 2)
            self._points_2d_camera[n] = RaySampler.Points2D(points_2d, camera_dims._camera_ids)


class UniformRaySampler(RaySampler):
    pass


class Rays:
    _origins: torch.Tensor
    _directions: torch.Tensor
    _lengths: torch.Tensor

    def __init__(self, cameras: PinholeCamera) -> None:
        pass

    def _calc_ray_params(self, cameras: PinholeCamera):
        pass
