import pytest
import torch

from kornia.geometry.camera import PinholeCamera
from kornia.nerf.nerf_model import MLP, NerfModel, NerfModelRenderer


@pytest.fixture
def nerf_model():
    return NerfModel(
        num_ray_points=11,
        num_pos_freqs=10,
        num_dir_freqs=4,
        num_units=2,
        num_unit_layers=4,
        num_hidden=256,
    )


def create_default_pinhole_camera(height, width, device, dtype):
    intrinsics = torch.eye(4, device=device, dtype=dtype)
    intrinsics[0, -1] = width / 2.0  # cx
    intrinsics[1, -1] = height / 2.0  # cy
    return PinholeCamera(
        intrinsics=intrinsics[None],
        extrinsics=torch.eye(4, device=device, dtype=dtype)[None],
        height=torch.tensor([height], device=device, dtype=dtype),
        width=torch.tensor([width], device=device, dtype=dtype),
    )


class TestNerfModel:
    def test_mlp(self, device, dtype):
        d_input = 63  # Input dimension after encoding
        num_hidden = 256
        mlp = MLP(d_input, num_units=2, num_unit_layers=4, num_hidden=num_hidden)
        mlp.to(device=device, dtype=dtype)

        num_rays = 15
        num_ray_points = 11
        x = torch.rand(num_rays, num_ray_points, d_input, device=device, dtype=dtype)
        xout = mlp(x)
        assert xout.shape == (num_rays, num_ray_points, num_hidden)

    def test_nerf(self, device, dtype, nerf_model):
        nerf_model = nerf_model.to(device=device, dtype=dtype)
        num_rays = 15
        origins = torch.rand(num_rays, 3, device=device, dtype=dtype)
        directions = torch.rand(num_rays, 3, device=device, dtype=dtype)
        rgbs = nerf_model(origins, directions)
        assert rgbs.shape == (num_rays, 3)

    def test_render_view(self, device, dtype, nerf_model):
        nerf_model = nerf_model.to(device=device, dtype=dtype)
        height, width = 5, 6
        renderer = NerfModelRenderer(nerf_model, image_size=(height, width), device=device, dtype=dtype)
        camera: PinholeCamera = create_default_pinhole_camera(height, width, device, dtype)
        image = renderer.render_view(camera)
        assert image.shape == (height, width, 3)
