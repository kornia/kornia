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

from kornia.geometry import solvers

from testing.base import BaseTester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rank3_matrix(null_vec: torch.Tensor, device, dtype) -> torch.Tensor:
    """Build a 3x4 matrix whose null space is spanned by *null_vec*.

    Strategy: start from the 4x4 identity, set the last row to *null_vec* so
    it becomes linearly dependent, then take the first three rows scaled by
    small random factors.  The result has rank 3 and its null vector is
    proportional to *null_vec*.
    """
    # Build the 4x4 matrix whose columns span the orthogonal complement of null_vec.
    # We use a Gram-Schmidt orthonormalisation relative to null_vec.
    n = null_vec.to(device=device, dtype=torch.float64)
    n = n / n.norm()

    # Start from the standard basis and remove the component along n.
    basis = torch.eye(4, device=device, dtype=torch.float64)
    vecs = []
    for i in range(4):
        v = basis[i] - (basis[i] @ n) * n
        if v.norm() > 1e-6:
            # Orthogonalise against already accepted vectors.
            for u in vecs:
                v = v - (v @ u) * u
            if v.norm() > 1e-6:
                vecs.append(v / v.norm())
        if len(vecs) == 3:
            break

    A = torch.stack(vecs, dim=0).to(dtype)  # (3, 4)
    return A.unsqueeze(0)  # (1, 3, 4)


class TestNullVector3x4(BaseTester):
    # ------------------------------------------------------------------
    # Smoke
    # ------------------------------------------------------------------

    def test_smoke(self, device, dtype):
        A = torch.rand(1, 3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.shape == (1, 4)

    # ------------------------------------------------------------------
    # Cardinality / shape
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch", [1, 2, 4, 8])
    def test_cardinality_batch(self, batch, device, dtype):
        A = torch.rand(batch, 3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.shape == (batch, 4)

    @pytest.mark.parametrize("extra", [(2, 3), (5,)])
    def test_cardinality_leading_dims(self, extra, device, dtype):
        A = torch.rand(*extra, 3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.shape == (*extra, 4)

    def test_cardinality_unbatched(self, device, dtype):
        A = torch.rand(3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.shape == (4,)

    # ------------------------------------------------------------------
    # Exception / input validation
    # ------------------------------------------------------------------

    def test_exception_wrong_rows(self, device, dtype):
        with pytest.raises(Exception):
            solvers.null_vector_3x4(torch.rand(1, 4, 4, device=device, dtype=dtype))

    def test_exception_wrong_cols(self, device, dtype):
        with pytest.raises(Exception):
            solvers.null_vector_3x4(torch.rand(1, 3, 3, device=device, dtype=dtype))

    def test_exception_1d(self, device, dtype):
        with pytest.raises(Exception):
            solvers.null_vector_3x4(torch.rand(4, device=device, dtype=dtype))

    def test_exception_not_tensor(self, device, dtype):
        with pytest.raises(Exception):
            solvers.null_vector_3x4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # ------------------------------------------------------------------
    # Known null vectors (hand-computed)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "A_data, expected_null",
        [
            # Standard basis: last column is free → null vector is e4 = [0,0,0,1].
            (
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]],
                [0.0, 0.0, 0.0, -1.0],  # sign comes from cofactor; direction only
            ),
            # A = [[1,0,0,1],[0,1,0,1],[0,0,1,1]] → null = [1,1,1,-1].
            (
                [[1.0, 0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0, 1.0]],
                [1.0, 1.0, 1.0, -1.0],
            ),
            # A = [[2,0,0,0],[0,3,0,0],[0,0,5,0]] → null = [0,0,0,1] (up to sign/scale).
            (
                [[2.0, 0.0, 0.0, 0.0],
                 [0.0, 3.0, 0.0, 0.0],
                 [0.0, 0.0, 5.0, 0.0]],
                [0.0, 0.0, 0.0, -30.0],
            ),
        ],
    )
    def test_known_null_vectors(self, A_data, expected_null, device, dtype):
        A = torch.tensor([A_data], device=device, dtype=dtype)          # (1, 3, 4)
        expected = torch.tensor([expected_null], device=device, dtype=dtype)  # (1, 4)
        v = solvers.null_vector_3x4(A)
        # Compare direction (ratio of corresponding components, ignoring global sign).
        # Normalise both and check |cos θ| ≈ 1.
        v_n = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        e_n = expected / expected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_theta = (v_n * e_n).sum(dim=-1).abs()
        self.assert_close(cos_theta, torch.ones_like(cos_theta), atol=1e-4, rtol=1e-4)

    # ------------------------------------------------------------------
    # Residual: A @ v must be (close to) zero
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "null_vec",
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -2.0, 3.0, -4.0],
            [0.5, 0.5, -0.5, 0.5],
        ],
    )
    def test_residual_near_zero(self, null_vec, device, dtype):
        """A @ null_vector_3x4(A) must be (close to) zero for any rank-3 A."""
        nv = torch.tensor(null_vec, dtype=torch.float64)
        A = _make_rank3_matrix(nv, device=device, dtype=dtype)  # (1, 3, 4)
        v = solvers.null_vector_3x4(A)                           # (1, 4)
        residual = (A @ v.unsqueeze(-1)).squeeze(-1)             # (1, 3)
        atol = {torch.float16: 1e-1, torch.bfloat16: 1e-1, torch.float32: 1e-4}.get(dtype, 1e-8)
        self.assert_close(residual, torch.zeros_like(residual), atol=atol, rtol=0.0)

    def test_residual_random_batch(self, device, dtype):
        """Batch of random rank-3 matrices: residual should vanish."""
        torch.manual_seed(42)
        B = 16
        null_vecs = torch.randn(B, 4)
        As = torch.cat([_make_rank3_matrix(null_vecs[i], device=device, dtype=dtype) for i in range(B)], dim=0)
        vs = solvers.null_vector_3x4(As)
        residuals = (As @ vs.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        atol = {torch.float16: 5e-1, torch.bfloat16: 5e-1, torch.float32: 1e-3}.get(dtype, 1e-7)
        self.assert_close(residuals, torch.zeros_like(residuals), atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Direction recovery: null vector matches the known one up to scale
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_direction_recovery(self, seed, device, dtype):
        """The returned vector must be collinear with the true null vector."""
        torch.manual_seed(seed)
        null_vec = torch.randn(4)
        A = _make_rank3_matrix(null_vec, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A).squeeze(0)  # (4,)

        v_n = v / v.norm().clamp(min=1e-8)
        nv_t = null_vec.to(device=device, dtype=dtype)
        nv_n = nv_t / nv_t.norm().clamp(min=1e-8)

        cos_theta = (v_n * nv_n).sum().abs()
        atol = {torch.float16: 1e-1, torch.bfloat16: 1e-1, torch.float32: 5e-4}.get(dtype, 1e-8)
        self.assert_close(cos_theta, cos_theta.new_ones(()), atol=atol, rtol=0.0)

    # ------------------------------------------------------------------
    # Dtype and device consistency
    # ------------------------------------------------------------------

    def test_output_dtype_matches_input(self, device, dtype):
        A = torch.rand(2, 3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.dtype == dtype

    def test_output_device_matches_input(self, device, dtype):
        A = torch.rand(2, 3, 4, device=device, dtype=dtype)
        v = solvers.null_vector_3x4(A)
        assert v.device == A.device

    # ------------------------------------------------------------------
    # Gradcheck
    # ------------------------------------------------------------------

    def test_gradcheck(self, device):
        # Use a well-conditioned rank-3 matrix to keep gradients stable.
        nv = torch.tensor([1.0, -2.0, 3.0, -4.0])
        A = _make_rank3_matrix(nv, device=device, dtype=torch.float64)
        A = A.detach().requires_grad_(True)
        self.gradcheck(solvers.null_vector_3x4, (A,), raise_exception=True, fast_mode=True)

    # ------------------------------------------------------------------
    # Dynamo / torch.compile
    # ------------------------------------------------------------------

    def test_dynamo(self, device, dtype, torch_optimizer):
        A = torch.rand(4, 3, 4, device=device, dtype=dtype)
        op = solvers.null_vector_3x4
        op_compiled = torch_optimizer(op)
        self.assert_close(op_compiled(A), op(A), atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Module (no nn.Module wrapper — use function call form)
    # ------------------------------------------------------------------

    def test_module(self, device, dtype):
        # null_vector_3x4 is a plain function; verify it is importable from the
        # top-level kornia.geometry.solvers namespace.
        import kornia.geometry.solvers as s
        assert hasattr(s, "null_vector_3x4")
        A = torch.rand(1, 3, 4, device=device, dtype=dtype)
        v = s.null_vector_3x4(A)
        assert v.shape == (1, 4)
