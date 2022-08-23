from typing import Dict, List, Optional, Tuple

import torch

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.geometry.nerf.types import Device
from kornia.utils.helpers import _torch_inverse_cast


def cameras_for_ids(cameras: PinholeCamera, camera_ids: List[int]):
    intrinsics = cameras.intrinsics[camera_ids]
    extrinsics = cameras.extrinsics[camera_ids]
    height = cameras.height[camera_ids]
    width = cameras.width[camera_ids]
    return PinholeCamera(intrinsics, extrinsics, height, width)


class RaySampler:
    _origins: Optional[torch.Tensor] = None  # Ray origins in world coordinates (*, 2)
    _directions: Optional[torch.Tensor] = None  # Ray directions in worlds coordinates (*, 2)
    _camera_ids: Optional[torch.Tensor] = None  # Ray camera ID
    _points_2d: Optional[torch.Tensor] = None  # Ray intersection with image plane in camera coordinates

    class Points2D_FlatTensors:
        def __init__(self) -> None:
            self._x: torch.Tensor
            self._y: torch.Tensor
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

    def __init__(self, min_depth: float, max_depth: float, device: Device) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._device = torch.device(device)

    @property
    def origins(self) -> torch.Tensor:
        return self._origins

    @property
    def directions(self) -> torch.Tensor:
        return self._directions

    @property
    def camera_ids(self) -> torch.Tensor:
        return self._camera_ids

    @property
    def points_2d(self) -> torch.Tensor:
        return self._points_2d

    def __len__(self) -> int:
        if self.origins is None:
            return 0
        return self.origins.shape[0]

    def _calc_ray_params(self, cameras: PinholeCamera, points_2d_camera: Dict[int, Points2D]) -> None:

        # Unproject 2d points in image plane to 3d world for two depths
        origins = []
        directions = []
        camera_ids = []
        points_2d = []
        for obj in points_2d_camera.values():
            num_cams_group, num_points_per_cam_group = obj._points_2d.shape[:2]
            depths = torch.ones(num_cams_group, 2 * num_points_per_cam_group, 3, device=self._device) * self._min_depth
            depths[:, num_points_per_cam_group:] = self._max_depth
            cams = cameras_for_ids(cameras, obj.camera_ids)
            points_3d = cams.unproject(obj._points_2d.repeat(1, 2, 1), depths)
            origins.append(points_3d[..., :num_points_per_cam_group, :].reshape(-1, 3))
            directions.append(
                (points_3d[..., num_points_per_cam_group:, :] - points_3d[..., :num_points_per_cam_group, :]).reshape(
                    -1, 3
                )
            )
            camera_ids.append(
                torch.tensor(obj.camera_ids).repeat(num_points_per_cam_group, 1).permute(1, 0).reshape(1, -1).squeeze(0)
            )
            points_2d.append(obj._points_2d.reshape(-1, 2).type(torch.uint8))
        self._origins = torch.cat(origins)
        self._directions = torch.cat(directions)
        self._camera_ids = torch.cat(camera_ids)
        self._points_2d = torch.cat(points_2d)

    def transform_ray_params_world_to_ndc(self, cameras: PinholeCamera) -> Tuple[torch.Tensor, torch.Tensor]:
        num_rays = self.__len__()
        lengths = sample_lengths(num_rays, 2, device=self._device, irregular=False)
        points_3d = sample_ray_points(self._origins, self._directions, lengths)
        cams = cameras_for_ids(cameras, self._camera_ids)
        points_3d_cams = cams.transform_to_camera_view(points_3d)

        # Camera to ndc projection matrix, assuming a symmetric viewing frustum
        H = torch.zeros((num_rays, 4, 4), device=self._device, dtype=torch.float32)
        fx = cams.fx
        fy = cams.fy
        widths = cams.width
        heights = cams.height
        H[..., 0, 0] = 2.0 * fx / widths
        H[..., 1, 1] = 2.0 * fy / heights
        H[..., 2, 2] = (self._max_depth + self._min_depth) / (self._max_depth - self._min_depth)
        H[..., 2, 3] = -self._max_depth * self._min_depth / (self._max_depth - self._min_depth)
        H[..., 3, 2] = 1.0
        points_3d_ndc = transform_points(H, points_3d_cams)

        R_inv = _torch_inverse_cast(cams.rotation_matrix)
        points_3d_ndc_world = (R_inv[:, None, ...].repeat(1, 2, 1, 1) @ points_3d_ndc[..., None]).squeeze()

        origins = points_3d_ndc_world[..., :1, :].squeeze()
        directions = (points_3d_ndc_world[..., :1, :] - points_3d_ndc_world[..., 1:, :]).squeeze()
        return origins, directions

    @staticmethod
    def _add_points2d_as_flat_tensors_to_num_ray_dict(
        n: int,
        x: torch.tensor,
        y: torch.tensor,
        camera_id: int,
        points2d_as_flat_tensors: Dict[int, Points2D_FlatTensors],
    ) -> None:
        if n not in points2d_as_flat_tensors:
            points2d_as_flat_tensors[n] = RaySampler.Points2D_FlatTensors()
            points2d_as_flat_tensors[n]._x = x.flatten()
            points2d_as_flat_tensors[n]._y = y.flatten()
        else:
            points2d_as_flat_tensors[n]._x = torch.cat((points2d_as_flat_tensors[n]._x, x.flatten()))
            points2d_as_flat_tensors[n]._y = torch.cat((points2d_as_flat_tensors[n]._y, y.flatten()))
        points2d_as_flat_tensors[n]._camera_ids.append(camera_id)

    @staticmethod
    def _build_num_ray_dict_of_points2d(
        points2d_as_flat_tensors: Dict[int, Points2D_FlatTensors]
    ) -> Dict[int, Points2D]:
        num_ray_dict_of_points2d: Dict[int, RaySampler.Points2D] = {}
        for n, points2d_as_list in points2d_as_flat_tensors.items():
            points_2d = (
                torch.stack((points2d_as_flat_tensors[n]._x, points2d_as_flat_tensors[n]._y))
                .permute(1, 0)
                .reshape(-1, n, 2)
            )
            num_ray_dict_of_points2d[n] = RaySampler.Points2D(points_2d, points2d_as_list._camera_ids)
        return num_ray_dict_of_points2d


class RandomRaySampler(RaySampler):
    def __init__(self, min_depth: float, max_depth: float, device: Device = 'cpu') -> None:
        super().__init__(min_depth, max_depth, device)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, num_img_rays: torch.Tensor
    ) -> Dict[int, RaySampler.Points2D]:
        num_img_rays = num_img_rays.int()
        points2d_as_flat_tensors: Dict[int, RaySampler.Points2D_FlatTensors] = {}
        for camera_id, (height, width, n) in enumerate(zip(heights.tolist(), widths.tolist(), num_img_rays.tolist())):
            y_rand = torch.trunc(torch.rand(n, device=self._device, dtype=torch.float32) * height)
            x_rand = torch.trunc(torch.rand(n, device=self._device, dtype=torch.float32) * width)
            RaySampler._add_points2d_as_flat_tensors_to_num_ray_dict(
                n, x_rand, y_rand, camera_id, points2d_as_flat_tensors
            )
        return RaySampler._build_num_ray_dict_of_points2d(points2d_as_flat_tensors)

    def calc_ray_params(self, cameras: PinholeCamera, num_img_rays: torch.Tensor) -> None:
        num_cams = cameras.batch_size
        if num_cams != num_img_rays.shape[0]:
            raise ValueError(
                'Number of cameras does not match size of tensor to define number of rays to march from each camera'
            )
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width, num_img_rays)
        self._calc_ray_params(cameras, points_2d_camera)


class UniformRaySampler(RaySampler):
    def __init__(self, min_depth: float, max_depth: float, device: Device = 'cpu') -> None:
        super().__init__(min_depth, max_depth, device)

    def sample_points_2d(self, heights: torch.Tensor, widths: torch.Tensor) -> Dict[int, RaySampler.Points2D]:
        heights = heights.int()
        widths = widths.int()
        points2d_as_flat_tensors: Dict[int, RaySampler.Points2D_FlatTensors] = {}
        for camera_id, (height, width) in enumerate(zip(heights.tolist(), widths.tolist())):
            n = height * width
            y_grid, x_grid = torch.meshgrid(
                torch.arange(height, device=self._device, dtype=torch.float32),
                torch.arange(width, device=self._device, dtype=torch.float32),
            )
            RaySampler._add_points2d_as_flat_tensors_to_num_ray_dict(
                n, x_grid, y_grid, camera_id, points2d_as_flat_tensors
            )
        return RaySampler._build_num_ray_dict_of_points2d(points2d_as_flat_tensors)

    def calc_ray_params(self, cameras: PinholeCamera) -> None:
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width)
        self._calc_ray_params(cameras, points_2d_camera)


def sample_lengths(num_rays: int, num_ray_points: int, device, irregular=False) -> torch.Tensor:
    if num_ray_points <= 1:
        raise ValueError('Number of ray points must be greater than 1')
    if not irregular:
        zero_to_one = torch.linspace(0.0, 1.0, num_ray_points, device=device)
        lengths = zero_to_one.repeat(num_rays, 1)  # FIXME: Expand instead of repeat maybe?
    else:
        zero_to_one = torch.linspace(0.0, 1.0, num_ray_points + 1, device=device)
        lengths = torch.rand(num_rays, num_ray_points, device=device) / num_ray_points + zero_to_one[:-1]
    return lengths


# TODO: Implement hierarchical ray sampling as described in Mildenhall (2020) Sec. 5.2


def sample_ray_points(
    origins: torch.Tensor, directions: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:  # FIXME: Test by projecting to points_2d and compare with sampler 2d points
    r"""
    Args:
        origins: tensor containing ray origins in 3d world coordinates. Tensor shape :math:`(*, 3)`.
        directions: tensor containing ray directions in 3d world coordinates. Tensor shape :math:`(*, 3)`.
        lengths: tensor cotaining sampled distances along each ray. Tensor shape :math:`(*, num_ray_points)`.

    Returns:
        points_3d: Points along rays :math:`(*, num_ray_points, 3)`
    """
    points_3d = origins[..., None, :] + lengths[..., None] * directions[..., None, :]
    return points_3d


def calc_ray_t_vals(points_3d: torch.Tensor) -> torch.Tensor:
    r"""Calculates t values along rays

    Args:
        points_3d: Points along rays :math:`(*, num_ray_points, 3)`

    Returns:
        t values along rays :math:`(*, num_ray_points)`

    Examples:       # FIXME: Fix this example!!
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    t_vals = torch.linalg.norm(points_3d - points_3d[..., 0, :].unsqueeze(-2), dim=-1)
    return t_vals
