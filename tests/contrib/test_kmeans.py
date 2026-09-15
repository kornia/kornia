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

from testing.base import BaseTester


class TestKMeans(BaseTester):
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
        from kornia.core.exceptions import BaseError, ShapeError

        # case: cluster_center = 0:
        with pytest.raises(BaseError) as errinfo:
            kornia.contrib.KMeans(0, None, 1e-3, 10, 0)
        assert "num_clusters can't be 0" in str(errinfo.value)

        # case: cluster centers is not a 2D tensor
        with pytest.raises(ShapeError) as errinfo:
            starting_centers = torch.rand((2, 3, 5), device=device, dtype=dtype)
            kmeans = kornia.contrib.KMeans(None, starting_centers, 1e-3, 100, 0)
        assert "Shape dimension mismatch" in str(errinfo.value)

        # case: input data is not a 2D tensor
        with pytest.raises(ShapeError) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 1e-3, 100, 0)
            kmeans.fit(torch.rand((1000, 5, 60), dtype=dtype, device=device))
        assert "Shape dimension mismatch" in str(errinfo.value) or "Expected shape" in str(errinfo.value)

        # case: column dimensions of cluster centers and data to be predicted do not match
        with pytest.raises(Exception) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 1e-3, 100, 0)
            kmeans.fit(torch.rand((1000, 5), dtype=dtype))
            kmeans.predict(torch.rand((10, 7), dtype=dtype))
        assert "7 != 5" in str(errinfo)

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
        self.assert_close(kmeans.cluster_centers, expected.to(device=device, dtype=dtype))

    @staticmethod
    def _create_data(device, dtype):
        # create example dataset
        torch.manual_seed(2023)
        x = 5 * torch.randn((500, 2), dtype=dtype, device=device) + torch.tensor((-13, 17), dtype=dtype, device=device)
        x = torch.vstack(
            [x, torch.randn((500, 2), dtype=dtype, device=device) + torch.tensor((15, -12), dtype=dtype, device=device)]
        )
        x = torch.vstack(
            [
                x,
                13 * torch.randn((500, 2), dtype=dtype, device=device)
                + torch.tensor((35, 15), dtype=dtype, device=device),
            ]
        )
        return x

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
