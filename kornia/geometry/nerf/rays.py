from typing import Dict, List  # , Tuple

import torch

from kornia.geometry.camera import PinholeCamera


def cameras_for_ids(cameras: PinholeCamera, camera_ids: List[int]):
    intrinsics = cameras.intrinsics[camera_ids]
    extrinsics = cameras.extrinsics[camera_ids]
    height = cameras.height[camera_ids]
    width = cameras.width[camera_ids]
    return PinholeCamera(intrinsics, extrinsics, height, width)


class RaySampler:
    _origins: torch.Tensor  # Ray origins in world coordinates
    _directions: torch.Tensor  # Ray directions in worlds coordinates
    _lengths: torch.Tensor  # Ray lengths
    _camera_ids: torch.Tensor  # Ray camera ID
    _points_2d: torch.Tensor  # Ray intersection with image plane in camera coordinates

    class Points2D_AsLists:
        def __init__(self) -> None:
            self._x: List[float] = []
            self._y: List[float] = []
            self._camera_ids: List[int] = []

    class Points2D:
        def __init__(self, points_2d: torch.Tensor, camera_ids: List[int]) -> None:
            self._points_2d = points_2d  # (*, N, 2)
            self._camera_ids = camera_ids

        @property
        def points_2d(self):
            return self._points_2d

        @property
        def camera_ids(self):
            return self._camera_ids

    def __init__(self, min_depth: float, max_depth: float, num_ray_points: int) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._num_ray_points = num_ray_points

    @property
    def origins(self) -> torch.Tensor:
        return self._origins

    @property
    def directions(self) -> torch.Tensor:
        return self._directions

    @property
    def lengths(self) -> torch.Tensor:
        return self._lengths

    def _calc_ray_params(self, cameras: PinholeCamera, points_2d_camera: Dict[int, Points2D]):

        # Unproject 2d points in image plane to 3d world for two depths
        origins = []
        directions = []
        lengths: List[torch.Tensor] = []
        for obj in points_2d_camera.values():
            num_cams_group, num_points_per_cam_group = obj._points_2d.shape[:2]
            num_points_group = num_cams_group * num_points_per_cam_group
            depths = torch.ones(num_cams_group, 2 * num_points_per_cam_group, 3) * self._min_depth
            depths[:, num_points_per_cam_group:] = self._max_depth
            cameras_id = cameras_for_ids(cameras, obj.camera_ids)
            points_3d = cameras_id.unproject(obj._points_2d.repeat(1, 2, 1), depths)
            origins.append(points_3d[..., :num_points_per_cam_group, :].reshape(-1, 3))
            directions.append(
                (points_3d[..., num_points_per_cam_group:, :] - points_3d[..., :num_points_per_cam_group, :]).reshape(
                    -1, 3
                )
            )
            lengths.append(torch.linspace(0.0, 1.0, self._num_ray_points).repeat(num_points_group, 1))
        self._origins = torch.cat(origins)
        self._directions = torch.cat(directions)
        self._lengths = torch.cat(lengths)

    @staticmethod
    def _add_points2d_as_lists_to_num_ray_dict(
        n: int, x: torch.tensor, y: torch.tensor, camera_id: int, points2d_as_lists: Dict[int, Points2D_AsLists]
    ) -> None:
        if n not in points2d_as_lists:
            points2d_as_lists[n] = RaySampler.Points2D_AsLists()
        points2d_as_lists[n]._x.extend(x.flatten().tolist())
        points2d_as_lists[n]._y.extend(y.flatten().tolist())
        points2d_as_lists[n]._camera_ids.append(camera_id)

    @staticmethod
    def _build_num_ray_dict_of_points2d(points2d_as_lists: Dict[int, Points2D_AsLists]) -> Dict[int, Points2D]:
        num_ray_dict_of_points2d: Dict[int, RaySampler.Points2D] = {}
        for n, points2d_as_list in points2d_as_lists.items():
            points_2d = (
                torch.stack((torch.tensor(points2d_as_lists[n]._x), torch.tensor(points2d_as_lists[n]._y)))
                .permute(1, 0)
                .reshape(-1, n, 2)
            )
            num_ray_dict_of_points2d[n] = RaySampler.Points2D(points_2d, points2d_as_list._camera_ids)
        return num_ray_dict_of_points2d


class RandomRaySampler(RaySampler):
    def __init__(self, min_depth: float, max_depth: float, num_ray_points: int) -> None:
        super().__init__(min_depth, max_depth, num_ray_points)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, num_rays: torch.Tensor
    ) -> Dict[int, RaySampler.Points2D]:
        num_rays = num_rays.int()
        points2d_as_lists: Dict[int, RaySampler.Points2D_AsLists] = {}
        for camera_id, (height, width, n) in enumerate(zip(heights.numpy(), widths.numpy(), num_rays.numpy())):
            y_rand = torch.trunc(torch.rand(n, dtype=torch.float32) * height)
            x_rand = torch.trunc(torch.rand(n, dtype=torch.float32) * width)
            RaySampler._add_points2d_as_lists_to_num_ray_dict(n, x_rand, y_rand, camera_id, points2d_as_lists)
        return RaySampler._build_num_ray_dict_of_points2d(points2d_as_lists)

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
        heights = heights.int()
        widths = widths.int()
        points2d_as_lists: Dict[int, RaySampler.Points2D_AsLists] = {}
        for camera_id, (height, width) in enumerate(zip(heights.numpy(), widths.numpy())):
            n = height * width
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32)
            )
            RaySampler._add_points2d_as_lists_to_num_ray_dict(n, x_grid, y_grid, camera_id, points2d_as_lists)
        return RaySampler._build_num_ray_dict_of_points2d(points2d_as_lists)

    def calc_ray_params(self, cameras: PinholeCamera):
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width)
        self._calc_ray_params(cameras, points_2d_camera)


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
