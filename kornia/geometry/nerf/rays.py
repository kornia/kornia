from typing import Dict, List  # , Tuple

import torch

from kornia.geometry.camera import PinholeCamera


class RaySampler:
    _origins: torch.Tensor  # Ray origins in world coordinates
    _directions: torch.Tensor  # Ray directions in worlds coordinates
    _lengths: torch.Tensor  # Ray lengths
    _camera_ids: torch.Tensor  # Ray camera ID
    _points_2d: torch.Tensor  # Ray intersection with image plane in camera coordinates

    class Points2D:
        def __init__(self, points_2d: torch.Tensor, camera_ids: List[int]) -> None:
            self._points_2d: torch.Tensor = points_2d
            self._camera_ids: torch.Tensor = camera_ids

        @property
        def points_2d(self):
            return self._points_2d

    def __init__(self, min_depth: float, max_depth: float, num_ray_points: int) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._num_ray_points = num_ray_points

    def _calc_ray_params(self, cameras: PinholeCamera, points_2d_camera: Dict[int, Points2D]):

        # Unproject 2d points in image plane to 3d world for two depths
        origins = []
        directions = []
        lengths: List[torch.Tensor] = []
        for obj in points_2d_camera.values():
            num_cams_group, num_points_per_cam_group = obj._points_2d.shape[:2]
            num_points_group = num_cams_group * num_points_per_cam_group
            depths = torch.ones(num_cams_group, 2 * num_points_per_cam_group, 3) * self.min_depth
            depths[:, num_points_per_cam_group:] = self.max_depth
            points_3d = cameras.unproject(obj._points_2d.repeat(1, 2, 1), depths).reshape(2 * num_points_group, -1)
            origins.append(points_3d[:num_points_group])
            directions.append(points_3d[:num_points_group] - points_3d[num_points_group:])
            lengths.append(
                torch.linspace(self.min_depth, self.max_depth, self.num_ray_points).repeat(num_points_group, 1)
            )
        self._origins = torch.cat(origins)
        self._directions = torch.cat(directions)  # FIXME: Directions should be normalized to unit vectors!
        self._lengths = torch.cat(lengths)


class RandomRaySampler(RaySampler):
    class CameraDims:
        def __init__(self) -> None:
            self._heights: List[int] = []
            self._widths: List[int] = []
            self._camera_ids: List[int] = []

    def __init__(self, min_depth: float, max_depth: float, num_ray_points: int) -> None:
        super().__init__(min_depth, max_depth, num_ray_points)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, num_rays: torch.Tensor
    ) -> Dict[int, RaySampler.Points2D]:
        num_rays: torch.Tensor = num_rays.int()
        cameras_dims: Dict[int, RandomRaySampler.CameraDims] = {}
        for camera_id, (height, width, n) in enumerate(zip(heights.numpy(), widths.numpy(), num_rays.numpy())):
            if n not in cameras_dims:
                cameras_dims[n] = RandomRaySampler.CameraDims()
            cameras_dims[n]._heights.extend([height] * n)
            cameras_dims[n]._widths.extend([width] * n)
            cameras_dims[n]._camera_ids.append(camera_id)

        points_2d_camera: Dict[int, RaySampler.Points2D] = {}
        for n, camera_dims in cameras_dims.items():
            dims = torch.tensor([camera_dims._heights, camera_dims._widths], dtype=torch.float32).T
            points_2d = torch.trunc(torch.rand_like(dims, dtype=torch.float32) * dims).reshape(-1, n, 2)
            points_2d_camera[n] = RaySampler.Points2D(points_2d, camera_dims._camera_ids)
        return points_2d_camera

    def calc_ray_params(self, cameras: PinholeCamera, num_rays: torch.Tensor):
        num_cams = cameras.height.shape[0]
        if num_cams != num_rays.shape[0]:
            raise ValueError(
                'Number of cameras does not match size of tensor to define number of rays to march from each camera'
            )
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width, num_rays)
        self._calc_ray_params(cameras, points_2d_camera)


class UniformRaySampler(RaySampler):
    def __init__(self, min_depth: float, max_depth: float, num_ray_points: int) -> None:
        super().__init__(min_depth, max_depth, num_ray_points)

    def sample_points_2d(self, heights: torch.Tensor, widths: torch.Tensor) -> Dict[int, RaySampler.Points2D]:
        points_2d: List[torch.Tensor] = []
        for height, width in zip(heights.numpy(), widths.numpy()):
            height_grid, width_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
            points_2d.append(torch.cat(height_grid, width_grid).reshape(-1, 2))
        num_cams = heights.shape[0]
        num_rays = points_2d.shape[0]
        return {num_rays: RaySampler.Points2D(torch.cat(points_2d), torch.arange(num_cams))}

    def calc_ray_params(self, cameras: PinholeCamera):
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width)
        self._calc_ray_params(cameras, points_2d_camera)


# class Rays:  # FIXME: This class should be merged with RaySampler above
#     _origins: torch.Tensor  # Ray origins in world coordinates
#     _directions: torch.Tensor  # Ray directions in worlds coordinates
#     _lengths: torch.Tensor  # Ray lengths
#     _camera_ids: torch.Tensor  # Ray camera ID
#     _points_2d: torch.Tensor  # Ray intersection with image plane in camera coordinates

#     # FIXME: Not division to cameras - just big tensors for each ray parameters

#     def __init__(
#         self,
#         cameras: PinholeCamera,
#         ray_sampler: RaySampler,
#         num_rays: torch.Tensor,
#         min_depth: float,
#         max_depth: float,
#         num_ray_points: int,
#     ) -> None:
#         num_cams = cameras.height.shape[0]
#         if num_cams != num_rays.shape[0]:
#             raise ValueError(
#                 'Number of cameras does not match size of tensor to define number of rays to march from each camera'
#             )
#         ray_sampler.sample_points_2d(cameras.height, cameras.width, num_rays)

#         # Unproject 2d points in image plane to 3d world for two depths
#         origins = []
#         directions = []
#         lengths: List[torch.Tensor] = []
#         for obj in ray_sampler.points_2d_camera.values():
#             num_cams_group, num_points_per_cam_group = obj.points_2d.shape[:2]
#             num_points_group = num_cams_group * num_points_per_cam_group
#             depths = torch.ones(num_cams_group, 2 * num_points_per_cam_group, 3) * min_depth
#             depths[:, num_points_per_cam_group:] = max_depth
#             points_3d = cameras.unproject(obj.points_2d.repeat(1, 2, 1), depths).reshape(2 * num_points_group, -1)
#             origins.append(points_3d[:num_points_group])
#             directions.append(points_3d[:num_points_group] - points_3d[num_points_group:])
#             lengths.append(torch.linspace(min_depth, max_depth, num_ray_points).repeat(num_points_group, 1))
#         self._origins = torch.cat(origins)
#         self._directions = torch.cat(directions)  # FIXME: Directions should be normalized to unit vectors!
#         self._legths = torch.cat(lengths)

#     def _calc_ray_params(self, cameras: PinholeCamera):
#         pass


def sample_ray_points(
    origins: torch.Tensor, directions: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:  # FIXME: Test by projecting to points_2d and compare with sampler 2d points
    r"""
    Args:
        origins: tensor containing ray origins in 3d world coordinates. Tensor shape :math:`(*, 3)`.
        directions: tensor containing ray directions in 3d world coordinates. Tensor shape :math:`(*, 3)`.
        lengths: tensor cotaining sampled distances along each ray. Tensor shape :math:`(*, num_ray_points)`.

    """
    points_3d = origins[..., None, :] + lengths[..., None] * directions[..., None, :]
    return points_3d
