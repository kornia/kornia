import math
from typing import Dict, List, Optional, Tuple

import torch

from kornia.geometry.camera import PinholeCamera
from kornia.nerf.camera_utils import cameras_for_ids
from kornia.nerf.types import Device
from kornia.utils.helpers import _torch_inverse_cast


class RaySampler:
    r"""Class to manage spatial ray sampling.

    Args:
        min_depth: sampled rays minimal depth from cameras: float
        max_depth: sampled rays maximal depth from cameras: float
        ndc: convert ray parameters to normalized device coordinates: bool
        device: device for ray tensors: Union[str, torch.device]
    """
    _origins: Optional[torch.Tensor] = None  # Ray origins in world coordinates (*, 3)
    _directions: Optional[torch.Tensor] = None  # Ray directions in world coordinates (*, 3)
    _directions_cam: Optional[torch.Tensor] = None  # Ray directions in camera coordinates (*, 3)
    _origins_cam: Optional[torch.Tensor] = None  # Ray origins in camera coordinates (*, 3)
    _camera_ids: Optional[torch.Tensor] = None  # Ray camera ID
    _points_2d: Optional[torch.Tensor] = None  # Ray intersection with image plane in camera coordinates

    def __init__(self, min_depth: float, max_depth: float, ndc: bool, device: Device) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ndc = ndc
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

    def _calc_ray_directions_cam(self, cameras: PinholeCamera, points_2d: torch.Tensor):
        # FIXME: This function should call perspective.unproject_points or, implement in PinholeCamera unproject to
        # camera coordinates that will call perspective.unproject_points
        fx = cameras.fx
        fy = cameras.fy
        cx = cameras.cx
        cy = cameras.cy
        directions_x = (points_2d[..., 0] - cx[..., None]) / fx[..., None]
        directions_y = (points_2d[..., 1] - cy[..., None]) / fy[..., None]
        directions_z = torch.ones_like(directions_x)
        directions_cam = torch.stack([directions_x, directions_y, directions_z], dim=-1)
        return directions_cam.reshape(-1, 3)

    class Points2D:
        r"""A class to hold ray 2d pixel coordinates and a camera id for each.

        Args:
            points_2d: tensor with ray pixel coordinates (the coordinates in the image plane that correspond to the
              ray):math:`(B, 2)`
            camera_ids: list of camera ids for each pixel coordinates: List[int]
        """

        def __init__(self, points_2d: torch.Tensor, camera_ids: List[int]) -> None:
            self._points_2d = points_2d  # (*, N, 2)
            self._camera_ids = camera_ids

        @property
        def points_2d(self):
            return self._points_2d

        @property
        def camera_ids(self):
            return self._camera_ids

    def _calc_ray_params(self, cameras: PinholeCamera, points_2d_camera: Dict[int, Points2D]) -> None:
        r"""Calculates ray parameters: origins, directions. Also stored are camera ids for each ray, and its pixel
        coordinates.

        Args:
            cameras: scene cameras: PinholeCamera
            points_2d_camera: a dictionary that groups Point2D objects by total number of casted rays
        """

        # Unproject 2d points in image plane to 3d world for two depths
        origins = []
        directions = []
        directions_cam = []
        origins_cam = []
        camera_ids = []
        points_2d = []
        for obj in points_2d_camera.values():

            # FIXME: Below both world and camera ray directions are calculated. It could be that world ray directions
            # will not be necessary and can be removed here
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
            directions_cam.append(self._calc_ray_directions_cam(cams, obj._points_2d))
            origins_cam.append(directions_cam[-1] * self._min_depth)
            camera_ids.append(
                torch.tensor(obj.camera_ids).repeat(num_points_per_cam_group, 1).permute(1, 0).reshape(1, -1).squeeze(0)
            )
            points_2d.append(obj._points_2d.reshape(-1, 2).int())
        self._origins = torch.cat(origins)
        self._directions = torch.cat(directions)
        self._directions_cam = torch.cat(directions_cam)
        self._origins_cam = torch.cat(origins_cam)
        self._camera_ids = torch.cat(camera_ids)
        if self._ndc:  # Transform ray parameters to NDC, if defined
            self._origins, self._directions = self.transform_ray_params_world_to_ndc(cameras)
        self._points_2d = torch.cat(points_2d)

    def transform_ray_params_world_to_ndc(self, cameras: PinholeCamera) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Transforms ray parameters to normalized coordinate device (camera) system (NDC)

        Args:
            cameras: scene cameras: PinholeCamera
        """
        cams = cameras_for_ids(cameras, self._camera_ids)
        fx = cams.fx
        fy = cams.fy
        widths = cams.width
        heights = cams.height
        fx_widths = 2.0 * fx / (widths - 1.0)
        fy_heights = 2.0 * fy / (heights - 1.0)

        # oxoz = self._origins_cam[..., 0] / self._origins_cam[..., 2]
        # oyoz = self._origins_cam[..., 1] / self._origins_cam[..., 2]

        oxoz = self._origins[..., 0] / self._origins[..., 2]
        oyoz = self._origins[..., 1] / self._origins[..., 2]

        origins_ndc_x = fx_widths * oxoz
        origins_ndc_y = fy_heights * oyoz

        # origins_ndc_z = 1 - 2 * self._min_depth / self._origins_cam[..., 2]

        origins_ndc_z = 1 - 2 * self._min_depth / self._origins[..., 2]

        origins_ndc = torch.stack([origins_ndc_x, origins_ndc_y, origins_ndc_z], dim=-1)

        # dxdz = self._directions_cam[..., 0] / self._directions_cam[..., 2]
        # dydz = self._directions_cam[..., 1] / self._directions_cam[..., 2]

        R_inv = _torch_inverse_cast(cams.rotation_matrix)
        directions_rotated_world = (R_inv @ self._directions_cam[..., None]).squeeze(dim=-1)

        dxdz = directions_rotated_world[..., 0] / directions_rotated_world[..., 2]
        dydz = directions_rotated_world[..., 1] / directions_rotated_world[..., 2]

        directions_ndc_x = fx_widths * dxdz - origins_ndc_x
        directions_ndc_y = fy_heights * dydz - origins_ndc_y
        directions_ndc_z = 1 - origins_ndc_z
        directions_ndc = torch.stack([directions_ndc_x, directions_ndc_y, directions_ndc_z], dim=-1)

        # R_inv = _torch_inverse_cast(cams.rotation_matrix)
        # origins_ndc_world = (R_inv @ origins_ndc[..., None]).squeeze(dim=-1)
        # directions_ndc_world = (R_inv @ directions_ndc[..., None]).squeeze(dim=-1)

        origins_ndc_world = origins_ndc
        directions_ndc_world = directions_ndc

        return origins_ndc_world, directions_ndc_world

        # FIXME: Remove or revise this part below
        # num_rays = self.__len__()
        # lengths = sample_lengths(num_rays, 2, device=self._device, irregular=False)
        # points_3d = sample_ray_points(self._origins, self._directions, lengths)
        # cams = cameras_for_ids(cameras, self._camera_ids)
        # points_3d_cams = cams.transform_to_camera_view(points_3d)

        # # Camera to ndc projection matrix, assuming a symmetric viewing frustum
        # H = torch.zeros((num_rays, 4, 4), device=self._device, dtype=torch.float32)  # self._max_depth->inf
        # fx = cams.fx
        # fy = cams.fy
        # widths = cams.width
        # heights = cams.height
        # H[..., 0, 0] = 2.0 * fx / widths
        # H[..., 1, 1] = 2.0 * fy / heights
        # H[..., 2, 2] = 1.0  # (self._max_depth + self._min_depth) / (self._max_depth - self._min_depth)
        # H[..., 2, 3] = (
        #     -2.0 * self._min_depth
        # )  # -2.0 * self._max_depth * self._min_depth / (self._max_depth - self._min_depth)
        # H[..., 3, 2] = 1.0
        # points_3d_ndc = transform_points(H, points_3d_cams)

        # # R_inv = _torch_inverse_cast(cams.rotation_matrix)  # FIXME: Not sure this is required for forward facing
        # cameras
        # # points_3d_ndc_world = (R_inv[:, None, ...].repeat(1, 2, 1, 1) @ points_3d_ndc[..., None]).squeeze(dim=-1)

        # # points_3d_ndc_world = cams.transform_to_world(points_3d_ndc)

        # points_3d_ndc_world = points_3d_ndc  # FIXME: Temp hack before asymptotic formulations for forward facing

        # origins = points_3d_ndc_world[..., :1, :].squeeze(dim=-2)
        # directions = (points_3d_ndc_world[..., :1, :] - points_3d_ndc_world[..., 1:, :]).squeeze(dim=-2)
        # return origins, directions

    class Points2D_FlatTensors:
        r"""Class to hold x/y pixel coordinates for each ray, and its scene camera id."""

        def __init__(self) -> None:
            self._x: torch.Tensor
            self._y: torch.Tensor
            self._camera_ids: List[int] = []

    @staticmethod
    def _add_points2d_as_flat_tensors_to_num_ray_dict(
        n: int,
        x: torch.tensor,
        y: torch.tensor,
        camera_id: int,
        points2d_as_flat_tensors: Dict[int, Points2D_FlatTensors],
    ) -> None:
        r"""Adds x/y pixel coordinates for all rays casted by a scene camera to dictionary of pixel coordinates
        grouped by total number of rays."""
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
        r"""Builds a dictionary of ray pixel points, by total number of rays as key. The dictionary groups rays by
        the total amount of rays, which allows the case of casting different number of rays from each scene camera.

        Args:
            points2d_as_flat_tensors: dictionary of pixel coordinates grouped by total number of rays:
              Dict[int, Points2D_FlatTensors]

        Returns:
            dictionary of Points2D objects that holds information on pixel 2d coordinates of each ray and the camera
              id it was casted by: Dict[int, Points2D]
        """
        num_ray_dict_of_points2d: Dict[int, RaySampler.Points2D] = {}
        for n, points2d_as_flat_tensor in points2d_as_flat_tensors.items():
            num_cams = len(points2d_as_flat_tensor._camera_ids)
            points_2d = (
                torch.stack((points2d_as_flat_tensor._x, points2d_as_flat_tensor._y))
                .permute(1, 0)
                .reshape(num_cams, -1, 2)
            )
            num_ray_dict_of_points2d[n] = RaySampler.Points2D(points_2d, points2d_as_flat_tensor._camera_ids)
        return num_ray_dict_of_points2d


class RandomRaySampler(RaySampler):
    r"""Class to manage random ray spatial sampling.

    Args:
        min_depth: sampled rays minimal depth from cameras: float
        max_depth: sampled rays maximal depth from cameras: float
        ndc: convert to normalized device coordinates: bool
        device: device for ray tensors: Union[str, torch.device]
    """

    def __init__(self, min_depth: float, max_depth: float, ndc: bool, device: Device = 'cpu') -> None:
        super().__init__(min_depth, max_depth, ndc, device)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, num_img_rays: torch.Tensor
    ) -> Dict[int, RaySampler.Points2D]:
        r"""Randomly sample pixel points in 2d.

        Args:
            heights: tensor that holds scene camera image heights (can vary between cameras): math: `(B)`.
            widths: tensor that holds scene camera image widths (can vary between cameras): math: `(B)`.
            num_img_rays: tensor that holds the number of rays to randomly cast from each scene camera: math: `(B)`.

        Returns:
            dictionary of Points2D objects that holds information on pixel 2d coordinates of each ray and the camera
              id it was casted by: Dict[int, Points2D]
        """
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
        r"""Calculates ray parameters: origins, directions. Also stored are camera ids for each ray, and its pixel
        coordinates.

        Args:
            cameras: scene cameras: PinholeCamera
            num_img_rays: tensor that holds the number of rays to randomly cast from each scene camera: int math: `(B)`.
        """
        num_cams = cameras.batch_size
        if num_cams != num_img_rays.shape[0]:
            raise ValueError(
                f'Number of cameras {num_cams} does not match size of tensor to define number of rays to march from '
                f'each camera {num_img_rays.shape[0]}'
            )
        points_2d_camera = self.sample_points_2d(cameras.height, cameras.width, num_img_rays)
        self._calc_ray_params(cameras, points_2d_camera)


class RandomGridRaySampler(RandomRaySampler):
    r"""Class to manage random ray spatial sampling. Sampling is done on a regular grid of pixels by randomizing
    column and row values, and casting rays for all pixels along the seleceted ones.

    Args:
        min_depth: sampled rays minimal depth from cameras: float
        max_depth: sampled rays maximal depth from cameras: float
        ndc: convert to normalized device coordinates: bool
        device: device for ray tensors: Union[str, torch.device]
    """

    def __init__(self, min_depth: float, max_depth: float, ndc: bool, device: Device = 'cpu') -> None:
        super().__init__(min_depth, max_depth, ndc, device)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, num_img_rays: torch.Tensor
    ) -> Dict[int, RaySampler.Points2D]:
        r"""Randomly sample pixel points in 2d over a regular row-column grid.

        Args:
            heights: tensor that holds scene camera image heights (can vary between cameras): math: `(B)`.
            widths: tensor that holds scene camera image widths (can vary between cameras): math: `(B)`.
            num_img_rays: tensor that holds the number of rays to randomly cast from each scene camera. Number of rows
              and columns is the square root of this value: int math: `(B)`.

        Returns:
            dictionary of Points2D objects that holds information on pixel 2d coordinates of each ray and the camera
              id it was casted by: Dict[int, Points2D]
        """
        num_img_rays = num_img_rays.int()
        points2d_as_flat_tensors: Dict[int, RaySampler.Points2D_FlatTensors] = {}
        for camera_id, (height, width, n) in enumerate(zip(heights.tolist(), widths.tolist(), num_img_rays.tolist())):
            n_sqrt = int(math.sqrt(n))
            y_rand = torch.randperm(int(height), device=self._device)[: min(int(height), n_sqrt)]
            x_rand = torch.randperm(int(width), device=self._device)[: min(int(width), n_sqrt)]
            y_grid, x_grid = torch.meshgrid(y_rand, x_rand, indexing='ij')
            RaySampler._add_points2d_as_flat_tensors_to_num_ray_dict(
                n_sqrt * n_sqrt, x_grid, y_grid, camera_id, points2d_as_flat_tensors
            )
        return RaySampler._build_num_ray_dict_of_points2d(points2d_as_flat_tensors)


class UniformRaySampler(RaySampler):
    r"""Class to manage uniform ray spatial sampling for all camera scene pixels.

    Args:
        min_depth: sampled rays minimal depth from cameras: float
        max_depth: sampled rays maximal depth from cameras: float
        ndc: convert to normalized device coordinates: bool
        device: device for ray tensors: Union[str, torch.device]
    """

    def __init__(self, min_depth: float, max_depth: float, ndc: bool, device: Device = 'cpu') -> None:
        super().__init__(min_depth, max_depth, ndc, device)

    def sample_points_2d(
        self, heights: torch.Tensor, widths: torch.Tensor, sampling_step=1
    ) -> Dict[int, RaySampler.Points2D]:
        r"""Uniformly sample pixel points in 2d for all scene camera pixels.

        Args:
            heights: tensor that holds scene camera image heights (can vary between cameras): math: `(B)`.
            widths: tensor that holds scene camera image widths (can vary between cameras): math: `(B)`.
            sampling_step: defines uniform strides between rows and columns: int.

        Returns:
            dictionary of Points2D objects that holds information on pixel 2d coordinates of each ray and the camera
              id it was casted by: Dict[int, Points2D]
        """
        heights = heights.int()
        widths = widths.int()
        points2d_as_flat_tensors: Dict[int, RaySampler.Points2D_FlatTensors] = {}
        for camera_id, (height, width) in enumerate(zip(heights.tolist(), widths.tolist())):
            n = height * width
            y_grid, x_grid = torch.meshgrid(
                torch.arange(0, height, sampling_step, device=self._device, dtype=torch.float32),
                torch.arange(0, width, sampling_step, device=self._device, dtype=torch.float32),
                indexing='ij',
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
    r"""Calculates t values along rays.

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
