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

from kornia.geometry.calibration.distort import distort_points, tilt_projection
from kornia.geometry.camera.distortion_affine import distort_points_affine

from testing.base import BaseTester
from testing.geometry.linalg import euler_angles_to_rotation_matrix


def _pz_r(taux, tauy, device, dtype):
    """Build OpenCV's ``(R, Pz)`` pair for a tilt of ``(taux, tauy)`` radians.

    Uses the independent Euler-angle test helper for ``R = Ry(tauy) @ Rx(taux)``;
    ``Pz`` follows OpenCV's ``modules/calib3d/src/distortion_model.hpp``.
    OpenCV's tilt projection is ``Pz @ R``; kornia returns ``Pz @ R.T`` (#4276).
    """
    taux = torch.tensor([taux], device=device, dtype=dtype)
    tauy = torch.tensor([tauy], device=device, dtype=dtype)
    one, zero = torch.ones_like(taux), torch.zeros_like(taux)
    r = euler_angles_to_rotation_matrix(-taux, -tauy, zero)[:, :3, :3]
    p_z = torch.stack(
        [r[..., 2, 2], zero, -r[..., 0, 2], zero, r[..., 2, 2], -r[..., 1, 2], zero, zero, one], -1
    ).reshape(-1, 3, 3)
    return r, p_z


def _k_asymmetric(device, dtype):
    """``fx = fy = 100``, ``cx = 4``, ``cy = 3`` -- ``cx != cy`` so a transposed reading changes the literals."""
    return torch.tensor([[[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)


class TestDistortPoints(BaseTester):
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 2, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(3, 3, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    def test_smoke_batch(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        pointsu = distort_points(points, K, distCoeff, new_K)
        assert points.shape == pointsu.shape

    @pytest.mark.parametrize(
        "batch_size, num_points, num_distcoeff", [(1, 3, 4), (2, 4, 5), (3, 5, 8), (4, 6, 12), (5, 7, 14)]
    )
    def test_shape(self, batch_size, num_points, num_distcoeff, device, dtype):
        B, N, Ndist = batch_size, num_points, num_distcoeff

        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(B, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(B, Ndist, device=device, dtype=dtype)

        pointsu = distort_points(points, K, distCoeff, new_K)
        assert pointsu.shape == (B, N, 2)

    def test_gradcheck(self, device):
        # ── ORIGINAL (partial): only points had requires_grad ──────────────
        # This left distCoeff, K, and new_K untested — meaning nobody had
        # verified that gradients actually flow through distortion coefficients.
        # That matters: without coefficient gradients you cannot optimise camera
        # intrinsics end-to-end via gradient descent (NeRF, bundle adjustment).
        #
        # ── FIX: enable requires_grad on ALL differentiable inputs ──────────
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_gradcheck_distcoeff_only(self, device):
        """Gradient flows through distortion coefficients independently.

        This is the critical use-case test: optimising distCoeff via gradient
        descent (e.g. camera calibration refinement during NeRF training) requires
        that d(output)/d(distCoeff) is non-zero and correctly computed.

        We fix points and K, and only differentiate through distCoeff — isolating
        the coefficient gradient path from the point gradient path.
        """
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64, requires_grad=True)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_gradcheck_K_only(self, device):
        """Gradient flows through the camera intrinsic matrix K independently.

        Verifies that d(output)/d(K) is correctly computed, which is required
        for joint optimisation of intrinsics and distortion parameters.
        """
        points = torch.rand(1, 8, 2, device=device, dtype=torch.float64)
        K = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        new_K = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(1, 4, device=device, dtype=torch.float64)

        assert self.gradcheck(distort_points, (points, K, distCoeff, new_K), raise_exception=True, fast_mode=True)

    def test_convention_accepts_4_5_8_12_14_coefficients_and_rejects_6(self, device, dtype):
        # Convention pin: the distortion vector is OpenCV's
        # (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, taux, tauy]]]]), so only the five prefix lengths
        # 4, 5, 8, 12 and 14 are meaningful; anything else -- 6 and 3 below -- is a ValueError, not a silent
        # zero-pad. Shorter accepted vectors ARE zero-padded to 14 internally, which is why 4 and 14 zeros give
        # the same answer. undistort_points enforces the identical rule.
        # Snippet used to generate expected: distort_points(pts, K, zeros(1, n)) for n in (3, 4, 5, 6, 8, 12, 14)
        # executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0, cpu and mps, every dtype) -> shape (1, 2, 2)
        # for the five accepted lengths, ValueError("Invalid number of distortion coefficients. Got 6") for 6;
        # torch.equal between the 4-coefficient answer and its 14-coefficient zero-padding -> True on every cell.
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        for n in (4, 5, 8, 12, 14):
            assert distort_points(points, K, torch.zeros(1, n, device=device, dtype=dtype)).shape == (1, 2, 2)
        for n in (3, 6):
            with pytest.raises(ValueError, match="Invalid number of distortion coefficients"):
                distort_points(points, K, torch.zeros(1, n, device=device, dtype=dtype))
        short = torch.tensor([[0.1, 0.01, 0.001, 0.001]], device=device, dtype=dtype)
        padded = torch.cat([short, torch.zeros(1, 10, device=device, dtype=dtype)], -1)
        assert not torch.allclose(distort_points(points, K, short).float(), points.float())
        assert torch.equal(distort_points(points, K, short), distort_points(points, K, padded))

    def test_convention_zero_coefficients_are_a_no_op_at_dtype_tolerance(self, device, dtype):
        # Convention pin: with every coefficient zero, distort_points normalizes
        # with new_K (= K here) and denormalizes with K, so it is a no-op -- but only up to the dtype tolerance,
        # because ``(u - cx) / fx`` followed by ``fx * x + cx`` is not an exact round trip for a general pixel.
        # This is the GENERAL statement; the bit-for-bit case is scoped in the sibling pin below.
        # The points sit half a focal length off the principal point on both sides of it, so the round trip is
        # exercised at a radius where a non-zero coefficient would move them by 2.775 px, and the fractional
        # offsets keep ``(u - cx) / fx`` off the exactly-representable grid.
        # Snippet used to generate expected: (distort_points(pts, K, zeros(1, 4)) - pts).abs().max() on
        # [[54.3, 53.7], [-16.1, 23.9]] executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0), differenced
        # in the working dtype -> cpu float32 3.81e-06 (torch.equal False), float64 0.0, float16 3.13e-02,
        # bfloat16 2.50e-01; mps float32 3.81e-06, float16 3.13e-02.
        points = torch.tensor([[[54.3, 53.7], [-16.1, 23.9]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        self.assert_close(distort_points(points, K, torch.zeros(1, 4, device=device, dtype=dtype)), points)

    def test_convention_zero_coefficients_are_bit_exact_on_these_points(self, device, dtype):
        # On these four coordinates the normalize/denormalize arithmetic rounds back to the input
        # bit for bit in the tested dtypes, even though +/-0.2 are not exactly representable in binary.
        # This is NOT a general bit-for-bit no-op. Executed counterexample on the same K, 2026-09-06 on the
        # commit c0b50ad7 (torch 2.14.0): the perturbed points [[54.3, 53.7], [-16.1, 23.9]] give torch.equal
        # False with a residual of 3.81e-06 in cpu float32 (3.13e-02 in float16, 2.50e-01 in bfloat16); only the
        # float64 cell is still exact there.
        # Snippet used to generate expected: torch.equal(distort_points(pts, K, zeros(1, 4)), pts) on
        # [[54.0, 53.0], [-16.0, 23.0]] executed 2026-09-06 -> True on cpu for float32/float64/float16/bfloat16
        # and on mps for float32/float16.
        points = torch.tensor([[[54.0, 53.0], [-16.0, 23.0]]], device=device, dtype=dtype)
        K = _k_asymmetric(device, dtype)
        assert torch.equal(distort_points(points, K, torch.zeros(1, 4, device=device, dtype=dtype)), points)

    def test_convention_new_K_normalizes_and_K_denormalizes(self, device, dtype):
        # Convention pin: the two intrinsics play OPPOSITE roles -- new_K maps
        # the incoming pixel to the normalized z = 1 plane, K maps the distorted normalized point back to pixels.
        # Every intrinsic is distinct here (new_K: fx 5, fy 7, cx 2, cy 4; K: fx 2, fy 3, cx 1, cy 1), so
        # swapping the two arguments, or fx with fy, changes both literals: (10, 10) normalizes to
        # ((10-2)/5, (10-4)/7) = (1.6, 6/7) and denormalizes to (2*1.6+1, 3*6/7+1) = (4.2, 3.571428...).
        # The diag(2, 2, 1) probe is the same claim with the roles reversed: a (1, 2) pixel
        # under new_K = diag(2, 2, 1) and K = eye(3) comes back as (0.5, 1.0).
        # Snippet used to generate expected: distort_points([[[10., 10.]]], K, zeros(1, 4), new_K) executed
        # 2026-09-06 on commit c0b50ad7 (torch 2.14.0, cpu float32) -> [[[4.199999809265137,
        # 3.5714285373687744]]]; the same call on mps float32 gives the same value.
        K = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        new_K = torch.tensor([[[5.0, 0.0, 2.0], [0.0, 7.0, 4.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        zero_dist = torch.zeros(1, 4, device=device, dtype=dtype)
        points = torch.tensor([[[10.0, 10.0]]], device=device, dtype=dtype)
        self.assert_close(
            distort_points(points, K, zero_dist, new_K),
            torch.tensor([[[4.2, 3.5714285714285716]]], device=device, dtype=dtype),
        )
        self.assert_close(
            distort_points(
                torch.tensor([[[1.0, 2.0]]], device=device, dtype=dtype),
                torch.eye(3, device=device, dtype=dtype)[None],
                zero_dist,
                torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype),
            ),
            torch.tensor([[[0.5, 1.0]]], device=device, dtype=dtype),
        )

    def test_convention_agrees_with_distort_points_affine_on_the_same_camera(self, device, dtype):
        # Convention pin: the two paths encode the same
        # pinhole camera through different INPUT DOMAINS and parametrizations. distort_points_affine takes a
        # normalized z = 1 point and a flat [fx, fy, cx, cy]; calibration.distort_points takes a pixel point, a
        # (3, 3) K and a coefficient vector. Feeding the normalized point through distort_points with
        # new_K = eye(3) (so the normalization step is the identity) and K = the same camera reproduces the
        # affine answer bit for bit, and the already-pixel point with zero coefficients
        # gives the same [54, 28].
        # Snippet used to generate expected: torch.equal(distort_points([[[0.5, 0.25]]], K, zeros(1, 4), eye(3)[None])
        # .reshape(2), distort_points_affine([0.5, 0.25], [100., 100., 4., 3.])) executed 2026-09-06 on the
        # commit c0b50ad7 (torch 2.14.0, cpu float32) -> True, value [54., 28.].
        K = _k_asymmetric(device, dtype)
        zero_dist = torch.zeros(1, 4, device=device, dtype=dtype)
        expected = torch.tensor([[[54.0, 28.0]]], device=device, dtype=dtype)
        affine = distort_points_affine(
            torch.tensor([0.5, 0.25], device=device, dtype=dtype),
            torch.tensor([100.0, 100.0, 4.0, 3.0], device=device, dtype=dtype),
        )
        self.assert_close(affine.reshape(1, 1, 2), expected, atol=0.0, rtol=0.0)
        self.assert_close(
            distort_points(
                torch.tensor([[[0.5, 0.25]]], device=device, dtype=dtype),
                K,
                zero_dist,
                torch.eye(3, device=device, dtype=dtype)[None],
            ),
            expected,
            atol=0.0,
            rtol=0.0,
        )
        self.assert_close(distort_points(expected, K, zero_dist), expected, atol=0.0, rtol=0.0)

    def test_convention_tilt_projection_zero_angles_are_the_identity(self, device, dtype):
        # Convention pin: tilt_projection(0, 0) is exactly eye(3) in BOTH branches,
        # which is why a caller who leaves the 13th and 14th coefficients at zero never meets kornia#4276 -- and
        # why the round trip pinned in test_undistort.py closes to 0.0 with tau = 0 and not with tau != 0.
        # Snippet used to generate expected: torch.equal(tilt_projection(tensor([0.]), tensor([0.])), eye(3)[None])
        # executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0) -> True on cpu for float32/float64/float16/
        # bfloat16 and on mps for float32/float16; same for return_inverse=True.
        zero = torch.zeros(1, device=device, dtype=dtype)
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        assert torch.equal(tilt_projection(zero, zero), identity)
        self.assert_close(tilt_projection(zero, zero, True), identity, atol=0.0, rtol=0.0)

    def test_convention_tilt_projection_inverse_branch_inverts_pz_times_r(self, device, dtype):
        # Convention pin: the ``return_inverse=True`` branch is the inverse of OpenCV's
        # ``Pz @ R``, not of kornia's own forward branch -- ``(Pz @ R) @ inverse`` is the identity at the dtype
        # tolerance, while ``(Pz @ R.T) @ inverse`` (kornia's forward reading, kornia#4276) is not.
        # This pin is PERMANENT and outlives the #4276 repair, and that is its point: the strict xfail below only
        # asks for ``forward @ inverse == I``, which the WRONG repair -- moving the inverse branch to
        # ``inv(Pz @ R.T)`` -- would satisfy just as well. #4276's Expected section forbids that repair because
        # undistort_points uses this branch and is what currently reproduces the repo's own cv2.undistortPoints
        # values (TestUndistortPoints::test_opencv_all_coeff in test_undistort.py). Written as a matrix product
        # rather than torch.linalg.inv so it runs on mps.
        # Snippet used to generate expected: ((Pz @ R) @ tilt_projection([0.1], [0.2], True) - eye(3)).abs().max()
        # executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0), differenced in the working dtype -> cpu
        # float32 1.19e-07, float64 2.22e-16, float16 4.88e-04, bfloat16 3.91e-03; mps float32 1.19e-07,
        # float16 4.88e-04. The same product built from Pz @ R.T is 0.396 away from the identity on every cell.
        # The second assertion uses a 0.1 SEPARATOR rather than ``not torch.allclose``: the identity's zero
        # entries pull torch.allclose's default atol down to 1e-08, which the correct reading's own float32
        # residual (1.19e-07) already exceeds, so ``not allclose`` would hold for BOTH readings and discriminate
        # nothing. 0.1 sits between the two populations (worst correct cell 3.91e-03, wrong reading 0.396).
        r, p_z = _pz_r(0.1, 0.2, device, dtype)
        inverse = tilt_projection(
            torch.tensor([0.1], device=device, dtype=dtype), torch.tensor([0.2], device=device, dtype=dtype), True
        )
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        self.assert_close((p_z @ r) @ inverse, identity)
        wrong = (p_z @ r.transpose(-1, -2)) @ inverse
        assert (wrong - identity).abs().max().item() > 0.1

    def test_wart_tilt_projection_forward_is_pz_times_r_transpose_4276(self, device, dtype):
        # Wart pin for kornia#4276: the forward branch returns ``Pz @ R.T``
        # where OpenCV's computeTiltProjectionMatrix returns ``Pz @ R``, while the return_inverse=True branch IS
        # the OpenCV inverse (pinned permanently by
        # test_convention_tilt_projection_inverse_branch_inverts_pz_times_r above). The consequence is that the two
        # branches of the same function are not inverses of each other: forward @ inverse is visibly not eye(3).
        # taux = 0.1 != tauy = 0.2, so Pz @ R and Pz @ R.T are different matrices here (they coincide at 0, 0).
        # Snippet used to generate expected: (tilt_projection([0.1], [0.2]) - Pz @ R.transpose(-1, -2)).abs().max()
        # executed 2026-09-06 on commit c0b50ad7 (torch 2.14.0) -> 0.0 on cpu for float32/float64/float16/
        # bfloat16 and on mps for float32/float16; forward @ inverse deviates from eye(3) by 0.3963 (cpu float32).
        # The last two assertions use a 0.1 separator for the same reason as the inverse-branch pin above:
        # ``not torch.allclose`` against an identity is satisfied by a 1e-07 residual too, so it would not flip
        # when #4276 is repaired. Pins the CURRENT value; NOT a contract; delete when #4276 is repaired.
        r, p_z = _pz_r(0.1, 0.2, device, dtype)
        taux = torch.tensor([0.1], device=device, dtype=dtype)
        tauy = torch.tensor([0.2], device=device, dtype=dtype)
        forward = tilt_projection(taux, tauy)
        self.assert_close(forward, p_z @ r.transpose(-1, -2), atol=0.0, rtol=0.0)
        assert (forward - p_z @ r).abs().max().item() > 0.1
        inverse = tilt_projection(taux, tauy, True)
        identity = torch.eye(3, device=device, dtype=dtype)[None]
        assert (forward @ inverse - identity).abs().max().item() > 0.1

    @pytest.mark.xfail(
        strict=True, reason="kornia#4276: the forward tilt branch returns Pz @ R.T, so it is not the inverse's inverse"
    )
    def test_convention_tilt_projection_branches_are_inverses_4276(self, device, dtype):
        # Intended contract, asserted as a strict xfail so the repair makes it XPASS and forces this mark out: the
        # two branches of tilt_projection document themselves as a matrix and its inverse
        # (``return_inverse``: "False to obtain the tilt projection matrix. True for the inverse matrix."), so
        # ``forward @ inverse`` must be the identity. Settled by the function's own signature and by #4276's
        # Expected section, which fixes the FORWARD branch and leaves the inverse branch -- the one that already
        # matches OpenCV and the repo's own cv2 pin -- untouched. The pin does not assert any particular OpenCV
        # form; #4276's ``Pz @ R`` claim comes from reading OpenCV's source, not from executing it here.
        taux = torch.tensor([0.1], device=device, dtype=dtype)
        tauy = torch.tensor([0.2], device=device, dtype=dtype)
        product = tilt_projection(taux, tauy) @ tilt_projection(taux, tauy, True)
        self.assert_close(product, torch.eye(3, device=device, dtype=dtype)[None])

    @pytest.mark.parametrize("return_inverse", [False, True])
    def test_wart_tilt_projection_flattens_leading_axes_4324(self, device, dtype, return_inverse):
        angles = torch.zeros(2, 3, 1, device=device, dtype=dtype)
        assert tilt_projection(angles, angles, return_inverse).shape == (6, 3, 3)

    def test_jit(self, device, dtype):
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        new_K = torch.rand(1, 3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(1, 4, device=device, dtype=dtype)
        inputs = (points, K, distCoeff, new_K)

        op = distort_points
        op_jit = torch.jit.script(op)
        self.assert_close(op(*inputs), op_jit(*inputs))
