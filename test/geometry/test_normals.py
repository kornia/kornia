import pytest
import torch
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils


def arguments(batch_size, device, dtype, s: int = 4):
    xx = torch.linspace(-1.0, 1.0, s, dtype=dtype, device=device).view(1, 1, 1, s).expand(1, 1, s, s)
    yy = torch.linspace(-1.0, 1.0, s, dtype=dtype, device=device).view(1, 1, s, 1).expand(1, 1, s, s)
    depth = xx ** 2 + yy ** 2
    camera_matrix = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).tile(batch_size, 1, 1)
    return depth, camera_matrix


_NORMAL_FROM_DEPTH_FUNC = {
    "normals_from_depth_accurate": kornia.geometry.normals_from_depth_accurate,
    "normals_from_depth_forward_gradient": kornia.geometry.normals_from_depth_forward_gradient,
    "normals_from_depth_improved": kornia.geometry.normals_from_depth_improved,
    "normals_from_depth_least_squares": kornia.geometry.normals_from_depth_least_squares,
}


class _TestNormalsFromDepth:
    normals_fn = None

    def test_smoke(self, device, dtype):
        depth, camera_matrix = arguments(1, device, dtype)
        normals = _NORMAL_FROM_DEPTH_FUNC[self.normals_fn](depth, camera_matrix)
        assert normals.shape == (1, 3, 4, 4)

    @pytest.mark.parametrize("batch_size", [2, 4, 5])
    def test_shapes(self, batch_size, device, dtype):
        depth, camera_matrix = arguments(batch_size, device, dtype)
        normals = _NORMAL_FROM_DEPTH_FUNC[self.normals_fn](depth, camera_matrix)
        assert normals.shape == (batch_size, 3, 4, 4)

    def test_gradcheck(self, device, dtype):
        depth, camera_matrix = arguments(1, device, dtype)
        depth = utils.tensor_to_gradcheck_var(depth)  # to var
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)  # to var
        assert gradcheck(_NORMAL_FROM_DEPTH_FUNC[self.normals_fn], (depth, camera_matrix), raise_exception=True)


class TestNormalsFromDepthAccurate(_TestNormalsFromDepth):
    normals_fn = "normals_from_depth_accurate"


class TestNormalsFromDepthForwardGradient(_TestNormalsFromDepth):
    normals_fn = "normals_from_depth_forward_gradient"


class TestNormalsFromDepthImproved(_TestNormalsFromDepth):
    normals_fn = "normals_from_depth_improved"


class TestNormalsFromDepthLeastSquares(_TestNormalsFromDepth):
    normals_fn = "normals_from_depth_least_squares"

    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_k(self, k, device, dtype):
        depth, camera_matrix = arguments(1, device, dtype, 16)
        normals = kornia.geometry.normals_from_depth_least_squares(depth, camera_matrix, k=k)
        assert normals.shape == (1, 3, 16, 16)

    @pytest.mark.parametrize("weighted", [True, False])
    def test_weighted(self, weighted, device, dtype):
        depth, camera_matrix = arguments(1, device, dtype, 16)
        normals = kornia.geometry.normals_from_depth_least_squares(depth, camera_matrix, weighted=weighted)
        assert normals.shape == (1, 3, 16, 16)

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
    def test_temperature(self, temperature, device, dtype):
        depth, camera_matrix = arguments(1, device, dtype, 16)
        normals = kornia.geometry.normals_from_depth_least_squares(depth, camera_matrix, temperature=temperature)
        assert normals.shape == (1, 3, 16, 16)
