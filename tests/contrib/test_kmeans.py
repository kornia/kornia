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

import kornia
from kornia.core.exceptions import BaseError, ShapeError

from testing.base import BaseTester


class TestKMeans(BaseTester):
    @pytest.mark.parametrize(
        ("tolerance", "max_iterations", "expected_centers", "expected_assignments"),
        [
            (None, 1, [[0.0], [6.0]], [0, 0, 1, 1, 1]),
            (100.0, 100, [[0.0], [6.0]], [0, 0, 1, 1, 1]),
            (1e-4, 100, [[2.0], [9.0]], [0, 0, 0, 1, 1]),
        ],
    )
    def test_final_assignments_match_centers(
        self, device, dtype, tolerance, max_iterations, expected_centers, expected_assignments
    ):
        x = torch.tensor([[0.0], [2.0], [4.0], [8.0], [10.0]], device=device, dtype=dtype)
        centers = torch.tensor([[0.0], [3.0]], device=device, dtype=dtype)
        kmeans = kornia.contrib.KMeans(2, centers, tolerance=tolerance, max_iterations=max_iterations)

        kmeans.fit(x)

        self.assert_close(kmeans.cluster_centers, x.new_tensor(expected_centers))
        expected = torch.tensor(expected_assignments, device=device, dtype=torch.long)
        self.assert_close(kmeans.cluster_assignments, expected)
        self.assert_close(kmeans.cluster_assignments, kmeans.predict(x))

    def test_final_assignments_after_refit(self, device, dtype):
        x = torch.tensor([[0.0], [2.0], [4.0], [8.0], [10.0]], device=device, dtype=dtype)
        centers = torch.tensor([[0.0], [3.0]], device=device, dtype=dtype)
        kmeans = kornia.contrib.KMeans(2, centers, tolerance=None, max_iterations=1)
        kmeans.fit(x)

        kmeans.fit(2 * x)

        self.assert_close(kmeans.cluster_centers, x.new_tensor([[0.0], [12.0]]))
        expected = torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.long)
        self.assert_close(kmeans.cluster_assignments, expected)
        self.assert_close(kmeans.cluster_assignments, kmeans.predict(2 * x))

    def test_dynamo_final_assignments(self, device, dtype, torch_optimizer):
        x = torch.tensor([[0.0], [2.0], [4.0], [8.0], [10.0]], device=device, dtype=dtype)
        centers = torch.tensor([[0.0], [3.0]], device=device, dtype=dtype)
        kmeans = kornia.contrib.KMeans(2, centers, tolerance=None, max_iterations=1)
        torch_optimizer(kmeans.fit)(x)

        expected = torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.long)
        self.assert_close(kmeans.cluster_assignments, expected)
        self.assert_close(kmeans.cluster_assignments, kmeans.predict(x))

    @pytest.mark.parametrize(
        ("num_clusters", "tolerance", "max_iterations"),
        [(3, 1e-4, 3), (10, 1e-3, 1000), (1, 1.0, 10)],
    )
    def test_smoke(self, device, dtype, num_clusters, tolerance, max_iterations):
        N = 1000
        D = 2

        kmeans = kornia.contrib.KMeans(num_clusters, None, tolerance, max_iterations, 0)
        kmeans.fit(torch.rand((N, D), dtype=dtype, device=device))

        out1 = kmeans.cluster_assignments
        out2 = kmeans.cluster_centers

        # output is of type tensor
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)

        # output is of same dtype
        assert out1.dtype == torch.int64
        assert out2.dtype == dtype

    @pytest.mark.parametrize("num_clusters", [3])
    @pytest.mark.parametrize("tolerance", [1e-3])
    @pytest.mark.parametrize("max_iterations", [100])
    def test_cardinality(self, device, dtype, num_clusters, tolerance, max_iterations):
        N = 1000
        D = 2

        kmeans = kornia.contrib.KMeans(num_clusters, None, tolerance, max_iterations, 0)
        kmeans.fit(torch.rand((N, D), device=device, dtype=dtype))

        out1 = kmeans.cluster_assignments
        out2 = kmeans.cluster_centers

        # output is of correct shape
        assert out1.shape == (N,)
        assert out2.shape == (num_clusters, D)

    def test_exception(self, device, dtype):
        # case: cluster_center = 0:
        with pytest.raises(BaseError) as errinfo:
            kornia.contrib.KMeans(0, None, 1e-3, 10, 0)
        assert "num_clusters can't be 0" in str(errinfo.value)

        # case: cluster centers is not a 2D tensor
        starting_centers = torch.rand((2, 3, 5), device=device, dtype=dtype)
        with pytest.raises(ShapeError) as errinfo:
            kornia.contrib.KMeans(None, starting_centers, 1e-3, 100, 0)
        assert "Shape dimension mismatch" in str(errinfo.value)

        # case: input data is not a 2D tensor
        kmeans = kornia.contrib.KMeans(3, None, 1e-3, 100, 0)
        with pytest.raises(ShapeError) as errinfo:
            kmeans.fit(torch.rand((1000, 5, 60), dtype=dtype, device=device))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        # case: column dimensions of cluster centers and data to be predicted do not match
        kmeans = kornia.contrib.KMeans(3, None, 1e-3, 100, 0)
        kmeans.fit(torch.rand((1000, 5), dtype=dtype))
        with pytest.raises(Exception) as errinfo:
            kmeans.predict(torch.rand((10, 7), dtype=dtype))
        assert "7 != 5" in str(errinfo)

        # case: num_clusters does not match the number of rows in an explicit cluster_centers
        starting_centers = torch.rand((5, 2), device=device, dtype=dtype)
        with pytest.raises(BaseError) as errinfo:
            kornia.contrib.KMeans(3, starting_centers, 1e-3, 100, 0)
        assert "cluster_centers has 5 rows but num_clusters=3" in str(errinfo.value)

    def test_empty_cluster_reseeds_to_a_data_point(self, device, dtype):
        # Both starting centers coincide, so every point ties to cluster 0 (argmin keeps the
        # first index on a tie) and cluster 1 gets no points assigned to it for the one update.
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device, dtype=dtype)
        starting_centers = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device, dtype=dtype)

        kmeans = kornia.contrib.KMeans(2, starting_centers, tolerance=None, max_iterations=1, seed=0)
        kmeans.fit(x)
        centers = kmeans.cluster_centers

        self.assert_close(centers[0], x.mean(dim=0))
        # the empty cluster is reseeded to *some* row of x, not left at the stale starting center
        assert any(torch.allclose(centers[1], row) for row in x)

    @pytest.mark.parametrize(
        ("x", "starting_centers", "expected"),
        [
            (
                (50 + torch.arange(3200) % 4).reshape(1600, 2).float(),
                None,
                torch.tensor([[51.0, 52.0]]),
            ),
            (
                torch.cat([torch.full((400, 3), 200.0), torch.full((400, 3), 20.0)]),
                torch.tensor([[190.0, 190.0, 190.0], [30.0, 30.0, 30.0]]),
                torch.tensor([[200.0, 200.0, 200.0], [20.0, 20.0, 20.0]]),
            ),
        ],
    )
    def test_large_cluster_sum_stays_finite(self, device, dtype, x, starting_centers, expected):
        # Regression test for accumulating sums/counts in X.dtype: in float16 a per-cluster sum
        # over hundreds of points routinely exceeds 65504 and overflows to inf; in bfloat16 the
        # count and sum both lose precision well before that. One update should still match the
        # exact float64 mean.
        x = x.to(device=device, dtype=dtype)
        centers = starting_centers.to(device=device, dtype=dtype) if starting_centers is not None else None
        num_clusters = expected.shape[0]

        kmeans = kornia.contrib.KMeans(num_clusters, centers, tolerance=None, max_iterations=1, seed=0)
        kmeans.fit(x)

        assert torch.isfinite(kmeans.cluster_centers).all()
        # exact in every dtype on the fixed implementation - a nonzero tolerance would let the
        # pre-fix bfloat16 error (0.25, well under BaseTester's default bfloat16 tolerance) pass
        self.assert_close(kmeans.cluster_centers, expected.to(device=device, dtype=dtype), rtol=0.0, atol=0.0)

    @staticmethod
    def _create_data(device, dtype):
        # create example dataset
        torch.manual_seed(2023)
        x = 5 * torch.randn((500, 2), dtype=dtype, device=device) + torch.tensor((-13, 17), dtype=dtype, device=device)
        x = torch.vstack(
            [x, torch.randn((500, 2), dtype=dtype, device=device) + torch.tensor((15, -12), dtype=dtype, device=device)]
        )
        return torch.vstack(
            [
                x,
                13 * torch.randn((500, 2), dtype=dtype, device=device)
                + torch.tensor((35, 15), dtype=dtype, device=device),
            ]
        )

    def test_module(self, device, dtype):
        x = TestKMeans._create_data(device, dtype)

        kmeans = kornia.contrib.KMeans(3, None, 1e-3, 10000, 2023)
        kmeans.fit(x)

        centers = kmeans.cluster_centers
        prediction = kmeans.predict(torch.tensor([[-14, 16], [45, 12]], dtype=dtype, device=device)).tolist()

        expected_centers = torch.tensor([[-13, 17], [15, -12], [35, 15]], dtype=dtype, device=device)
        expected_prediction = [0, 2]

        # sorting centers using dimension 0 as key so that they can be checked for equalness
        order = torch.argsort(centers[:, 0]).tolist()
        new_classes = {old_class: new_class for new_class, old_class in enumerate(order)}

        ordered_centers = centers[order]
        oredered_prediction = [new_classes[predicted_class] for predicted_class in prediction]

        self.assert_close(ordered_centers, expected_centers, atol=2, rtol=0.1)
        assert oredered_prediction == expected_prediction

    def test_dynamo(self, device, dtype, torch_optimizer):
        x = TestKMeans._create_data(device, dtype)
        kmeans_params = (3, None, 1e-3, 10000, 2023)
        predict_param = torch.tensor([[-14, 16], [45, 12]], dtype=dtype, device=device)

        kmeans = kornia.contrib.KMeans(*kmeans_params)
        kmeans.fit(x)

        centers = kmeans.cluster_centers
        prediction = kmeans.predict(predict_param)

        kmeans_op = kornia.contrib.KMeans(*kmeans_params)
        kmeans_op.fit = torch_optimizer(kmeans_op.fit)
        kmeans_op.predict = torch_optimizer(kmeans_op.predict)

        kmeans_op.fit(x)

        centers_op = kmeans_op.cluster_centers
        prediction_op = kmeans_op.predict(predict_param)

        self.assert_close(centers, centers_op)
        self.assert_close(prediction, prediction_op)

    def _blobs(self, device, dtype):
        g = torch.Generator().manual_seed(0)
        return torch.cat(
            [torch.randn(60, 2, generator=g) + torch.tensor(c) for c in ([0.0, 0.0], [6.0, 0.0], [0.0, 6.0])]
        ).to(device, dtype)

    def test_zero_tolerance_stops_at_the_fixed_point(self, device, dtype):
        # tolerance=0.0 with the default max_iterations=0 never returned.
        X = self._blobs(device, dtype)
        km = kornia.contrib.KMeans(3, X[[0, 60, 120]].clone(), tolerance=0.0)
        calls = {"n": 0}
        original = km._pairwise_euclidean_distance

        def counting(a, b):
            calls["n"] += 1
            if calls["n"] > 200:
                raise RuntimeError("fit did not stop")
            return original(a, b)

        km._pairwise_euclidean_distance = counting
        km.fit(X)
        assert calls["n"] < 50

    def test_unbounded_configuration_raises(self):
        with pytest.raises(BaseError, match="max_iterations"):
            kornia.contrib.KMeans(3, None, tolerance=None, max_iterations=0)
        with pytest.raises(BaseError, match="tolerance"):
            kornia.contrib.KMeans(3, None, tolerance=-1.0)

    def test_refit_samples_fresh_initial_centres(self, device, dtype):
        # A second fit reused the first fit's sampled centres (and raised for another dimension).
        km = kornia.contrib.KMeans(3, None, max_iterations=5, seed=0)
        km.fit(torch.rand(180, 2, device=device, dtype=dtype))
        X5 = torch.rand(100, 5, device=device, dtype=dtype) + 100
        km.fit(X5)
        assert km.cluster_centers.shape == (3, 5)
        assert (km.cluster_centers >= 100).all()

    def test_float16_distances_do_not_overflow(self, device):
        # float16 squared distances overflowed to inf, so argmin returned [0, 0]; float64 gives [1, 0].
        c = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], device=device, dtype=torch.float16)
        p = torch.tensor([[255.0, 255.0, 255.0], [10.0, 200.0, 190.0]], device=device, dtype=torch.float16)
        assert kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1).predict(p).tolist() == [1, 0]

    def test_float16_fit_distances_do_not_overflow(self, device):
        # The same float16 overflow in fit sent both points to cluster 0 and reseeded cluster 1 at random.
        c = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], device=device, dtype=torch.float16)
        p = torch.tensor([[255.0, 255.0, 255.0], [10.0, 200.0, 190.0]], device=device, dtype=torch.float16)
        km = kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1)
        km.fit(p)
        self.assert_close(km.cluster_centers, p[[1, 0]], rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("pixel_dtype", [torch.uint8, torch.int64])
    def test_predict_integer_input_with_float_centres(self, device, pixel_dtype):
        # Colour quantisation: fit on float pixels, predict integer pixels. The float centres must not be cast to the
        # integer dtype (uint8 wraps, int64 truncates). Reference: torch.cdist in float64 on the CPU.
        g = torch.Generator().manual_seed(0)
        Xf = (torch.rand(500, 3, generator=g) * 255).round()
        centres = Xf[:8].clone()
        km = kornia.contrib.KMeans(8, centres.to(device), tolerance=1e-4, max_iterations=50)
        km.fit(Xf.to(device))
        expected = torch.cdist(Xf.double(), km.cluster_centers.cpu().double()).argmin(-1)
        got = km.predict(Xf.to(pixel_dtype).to(device)).cpu()
        assert torch.equal(got, expected)

    @pytest.mark.parametrize("pixel_dtype", [torch.uint8, torch.int64])
    def test_predict_does_not_truncate_the_float_centres(self, device, pixel_dtype):
        # Casting the centres to the pixel dtype truncates 0.9 to 0 and ties the two centres (argmin keeps index 0).
        c = torch.tensor([[0.0, 0.0], [0.9, 0.9]], device=device)
        p = torch.tensor([[1, 1]], dtype=pixel_dtype, device=device)
        assert kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1).predict(p).tolist() == [1]

    def test_uint8_centres_and_points_are_compared_without_wrapping(self, device):
        # uint8 differences and squares wrap: in uint8 the pixel (40, 40) looks nearest to (200, 200), but (0, 0) is.
        c = torch.tensor([[0, 0], [200, 200]], dtype=torch.uint8, device=device)
        p = torch.tensor([[40, 40]], dtype=torch.uint8, device=device)
        assert kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1).predict(p).tolist() == [0]

    def test_fit_uint8_points_with_float_centres_does_not_wrap_the_centres(self, device):
        c = torch.tensor([[300.0, 300.0], [100.0, 100.0]], device=device)
        p = torch.tensor([[250, 250], [90, 90]], dtype=torch.uint8, device=device)
        km = kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1)
        km.fit(p)
        assert km.cluster_centers.tolist() == [[250, 250], [90, 90]]

    def test_fit_sums_uint8_points_without_wrapping(self, device):
        # uint8 sums wrap: the mean of 200 and 210 came out as 77. Two empty-cluster-free clusters, one update.
        c = torch.tensor([[205.0, 205.0], [5.0, 5.0]], device=device)
        p = torch.tensor([[200, 200], [210, 210], [4, 4]], dtype=torch.uint8, device=device)
        km = kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1)
        km.fit(p)
        assert km.cluster_centers.tolist() == [[205, 205], [4, 4]]

    def test_predict_int64_values_above_2_pow_24_stay_exact(self, device):
        # Integer data against integer centres is compared in int64: float32 maps both centres to 16777216.
        c = torch.tensor([[16777216], [16777217]], dtype=torch.int64, device=device)
        x = torch.tensor([[16777217]], dtype=torch.int64, device=device)
        assert kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1).predict(x).tolist() == [1]

    def test_predict_integer_squared_distances_are_not_rounded_through_float32_sqrt(self, device):
        # Squared distances 10**8 and 10**8 + 1 have the same float32 square root; the nearest centre is index 1.
        c = torch.tensor([[10000, 1], [10000, 0]], dtype=torch.int64, device=device)
        x = torch.tensor([[0, 0]], dtype=torch.int64, device=device)
        assert kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1).predict(x).tolist() == [1]

    def test_fit_int64_mean_above_2_pow_24_stays_exact(self, device):
        # The sum 33554434 is not a float32 value (it rounds to 33554432), so a float mean would give 16777216.
        probe = torch.div(
            torch.tensor([33554434], device=device), torch.tensor([2], device=device), rounding_mode="trunc"
        )
        if probe.item() != 16777217:
            pytest.skip("this torch/device computes int64 trunc division through float32 (MPS, torch 2.5)")
        x = torch.tensor([[16777217], [16777217], [5]], dtype=torch.int64, device=device)
        c = torch.tensor([[16777217], [5]], dtype=torch.int64, device=device)
        km = kornia.contrib.KMeans(2, c, tolerance=None, max_iterations=1)
        km.fit(x)
        assert km.cluster_centers.tolist() == [[16777217], [5]]

    def test_integer_zero_tolerance_shift_does_not_wrap(self, device):
        # The first update gives [[16], [208]]; its squared shifts (16**2, 80**2) are 0 mod 256, so a uint8 shift
        # stopped the fit there, although the next update gives [[37], [240]].
        x = torch.tensor([[0], [32], [80], [240], [240], [240], [240]], dtype=torch.uint8, device=device)
        c = torch.tensor([[0], [128]], dtype=torch.uint8, device=device)
        km = kornia.contrib.KMeans(2, c, tolerance=0.0)
        km.fit(x)
        assert km.cluster_centers.tolist() == [[37], [240]]

    def test_negative_max_iterations_raises(self):
        with pytest.raises(BaseError, match="max_iterations must be >= 0"):
            kornia.contrib.KMeans(3, None, tolerance=None, max_iterations=-1)
        with pytest.raises(BaseError, match="max_iterations must be >= 0"):
            kornia.contrib.KMeans(3, None, tolerance=1e-3, max_iterations=-2)

    def test_predict_checks_its_input_rank(self, device, dtype):
        km = kornia.contrib.KMeans(2, torch.rand(2, 2, device=device, dtype=dtype), max_iterations=1)
        with pytest.raises(ShapeError):
            km.predict(torch.rand(4, 2, 1, device=device, dtype=dtype))
