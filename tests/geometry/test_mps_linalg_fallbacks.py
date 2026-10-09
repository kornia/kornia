# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

from kornia.geometry import transform_points
from kornia.geometry.calibration.pnp import solve_pnp_dlt
from kornia.geometry.homography import find_homography_dlt
from kornia.geometry.transform import get_perspective_transform3d, get_tps_transform


def _reject_mps(fn):
    def wrapped(*args, **kwargs):
        if any(torch.is_tensor(arg) and arg.device.type == "mps" for arg in args):
            raise NotImplementedError("MPS kernel unavailable")
        return fn(*args, **kwargs)

    return wrapped


def test_mps_geometry_helpers_fall_back_to_cpu(monkeypatch):
    """Public geometry APIs on MPS must avoid kernels missing on the torch floor."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    for name in ("lu_factor_ex", "lu_solve", "solve", "solve_ex", "svdvals", "qr"):
        monkeypatch.setattr(torch.linalg, name, _reject_mps(getattr(torch.linalg, name)))
    monkeypatch.setattr(torch, "lu_unpack", _reject_mps(torch.lu_unpack))
    monkeypatch.setattr(torch, "det", _reject_mps(torch.det))

    device = torch.device("mps")
    torch.manual_seed(0)

    points1 = torch.rand(1, 8, 2, device=device)
    homography = torch.tensor([[[1.0, 0.1, 0.2], [0.0, 1.1, 0.1], [0.001, 0.0, 1.0]]], device=device)
    points2 = transform_points(homography, points1)
    estimated_homography = find_homography_dlt(points1, points2, solver="lu")

    world_points = torch.rand(1, 6, 3, device=device) + torch.tensor([0.0, 0.0, 5.0], device=device)
    image_points = torch.rand(1, 6, 2, device=device)
    intrinsics = torch.eye(3, device=device)[None]
    world_to_camera = solve_pnp_dlt(world_points, image_points, intrinsics)

    tps_source = torch.rand(1, 5, 2, device=device)
    tps_destination = torch.rand(1, 5, 2, device=device)
    tps = get_tps_transform(tps_source, tps_destination)

    source3d = torch.rand(1, 8, 3, device=device)
    destination3d = torch.rand(1, 8, 3, device=device)
    perspective3d = get_perspective_transform3d(source3d, destination3d)

    results = []
    for result in (estimated_homography, world_to_camera, tps, perspective3d):
        results.extend(result if isinstance(result, tuple) else (result,))
    for result in results:
        assert result.device.type == "mps"
        assert torch.isfinite(result).all()
