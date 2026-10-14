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

# based on https://github.com/subhadarship/kmeans_pytorch

from __future__ import annotations

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import euclidean_distance


class KMeans:
    """Implements the kmeans clustering algorithm with euclidean distance as similarity measure.

    Args:
        num_clusters: number of clusters the data has to be assigned to
        cluster_centers: torch.Tensor of starting cluster centres can be passed instead of num_clusters
        tolerance: float value. the algorithm terminates if the shift in centers is less than tolerance
        max_iterations: number of iterations to run the algorithm for
        seed: number to set torch manual seed for reproducibility

    Example:
        >>> kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, 0)
        >>> kmeans.fit(torch.rand((1000, 5)))
        >>> predictions = kmeans.predict(torch.rand((10, 5)))

    """

    def __init__(
        self,
        num_clusters: int,
        cluster_centers: torch.Tensor | None,
        tolerance: float = 10e-4,
        max_iterations: int = 0,
        seed: int | None = None,
    ) -> None:
        KORNIA_CHECK(num_clusters != 0, "num_clusters can't be 0")
        KORNIA_CHECK(max_iterations >= 0, f"max_iterations must be >= 0, got {max_iterations}")
        KORNIA_CHECK(tolerance is None or tolerance >= 0, f"tolerance must be None or >= 0, got {tolerance}")
        KORNIA_CHECK(
            tolerance is not None or max_iterations > 0,
            "tolerance=None needs max_iterations > 0: with max_iterations=0 the fit would never stop.",
        )

        # cluster_centers should have only 2 dimensions
        if cluster_centers is not None:
            KORNIA_CHECK_SHAPE(cluster_centers, ["C", "D"])
            KORNIA_CHECK(
                cluster_centers.shape[0] == num_clusters,
                f"cluster_centers has {cluster_centers.shape[0]} rows but num_clusters={num_clusters}",
            )

        self.num_clusters = num_clusters
        self._initial_cluster_centers = cluster_centers  # user-given start; None means sample from each fit's X
        self._cluster_centers = cluster_centers
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self._final_cluster_assignments: torch.Tensor | None = None
        self._final_cluster_centers: torch.Tensor | None = None

        if seed is not None:
            torch.manual_seed(seed)

    @property
    def cluster_centers(self) -> torch.Tensor:
        """Return the current cluster centers.

        Returns:
            A tensor with shape :math:`(C, D)`:
            - ``C`` is the number of clusters.
            - ``D`` is the feature dimension of each sample.

            If :meth:`fit` has already been called, this returns the learned
            final centers. Otherwise, it returns the initialization provided
            during construction.

        Raises:
            TypeError: If no initial centers were provided and ``fit`` has not been run.
        """
        if isinstance(self._final_cluster_centers, torch.Tensor):
            return self._final_cluster_centers
        if isinstance(self._cluster_centers, torch.Tensor):
            return self._cluster_centers
        raise TypeError("Model has not been fit to a dataset")

    @property
    def cluster_assignments(self) -> torch.Tensor:
        """Return cluster labels assigned during the most recent ``fit`` call.

        Returns:
            A 1D tensor with shape :math:`(N,)`, where ``N`` is the number of
            samples given to :meth:`fit`. Each value is the cluster index
            nearest to the corresponding sample in the final centers, matching
            :meth:`predict` on the training data even when fitting stops early.

        Raises:
            TypeError: If ``fit`` has not been run yet.
        """
        if isinstance(self._final_cluster_assignments, torch.Tensor):
            return self._final_cluster_assignments
        raise TypeError("Model has not been fit to a dataset")

    @staticmethod
    def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
        """Dtype for distances and sums: the promoted dtype.

        Half types lift to float32. When every tensor is integer the work dtype is int64, which is exact (float32
        would merge integers above 2**24, and uint8 arithmetic wraps); squared differences of values within +-2**31
        fit in int64.
        """
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if not dtype.is_floating_point:
            return torch.int64
        return dtype

    def _initialise_cluster_centers(self, X: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Chooses num_cluster points from X as the initial cluster centers.

        Args:
            X: 2D input torch.Tensor to be clustered
            num_clusters: number of desired cluster centers

        Returns:
            2D torch.Tensor with num_cluster rows

        """
        num_samples: int = len(X)
        perm = torch.randperm(num_samples, device=X.device)
        idx = perm[:num_clusters]
        return X[idx]

    def _pairwise_euclidean_distance(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared distance between 2 sets of vectors.

        Args:
            data1: 2D torch.Tensor of shape N, D
            data2: 2D torch.Tensor of shape C, D

        Returns:
            2D torch.Tensor of shape N, C

        """
        # N*1*D
        A = data1[:, None, ...]
        # 1*C*D
        B = data2[None, ...]
        if not A.dtype.is_floating_point and not B.dtype.is_floating_point:
            # Integer vectors: the squared distance is exact in int64, whereas its square root would round to float32.
            # The caller only takes an argmin, which the square root does not change.
            return (A - B).pow(2).sum(dim=-1)
        return euclidean_distance(A, B)

    def fit(self, X: torch.Tensor) -> None:
        """Fit iterative KMeans clustering till a threshold for shift in cluster centers or a maximum no of iterations
        have reached.

        Args:
            X: 2D input torch.Tensor to be clustered

        """  # noqa: D205
        # X should have only 2 dimensions
        KORNIA_CHECK_SHAPE(X, ["N", "D"])

        if self._initial_cluster_centers is None:
            current_centers = self._initialise_cluster_centers(X, self.num_clusters)
        else:
            # X and cluster_centers should have same number of columns
            KORNIA_CHECK(
                X.shape[1] == self._initial_cluster_centers.shape[1],
                f"Dimensions at position 1 of X and cluster_centers do not match. \
                {X.shape[1]} != {self._initial_cluster_centers.shape[1]}",
            )
            current_centers = self._initial_cluster_centers

        # Distances and sums run in at least float32: squared float16 distances overflow past 255.
        work_dtype = self._work_dtype(X, current_centers)
        X_work = X.to(work_dtype)

        previous_centers: torch.Tensor | None = None
        iteration: int = 0

        while True:
            # find distance between X and current_centers
            distance: torch.Tensor = self._pairwise_euclidean_distance(X_work, current_centers.to(work_dtype))

            cluster_assignment = distance.argmin(-1)

            previous_centers = current_centers.clone()

            # Vectorized per-cluster mean via a one-hot assignment matrix, instead of looping over
            # clusters with torch.nonzero/index_select (each iteration's dynamic output shape
            # forces a host sync). One-hot + matmul routes the reduction through a GEMM kernel;
            # scatter_add_ into the small (num_clusters, D) destination measured ~10x slower on
            # MPS for this shape (many threads racing to accumulate into few destination rows).
            #
            # Counts stay int64 (exact) and the sum accumulates in at least float32: X.dtype
            # itself overflows to inf in float16 past 65504, and loses precision in bfloat16,
            # for perfectly ordinary inputs (e.g. float16 image color quantization).
            one_hot = torch.nn.functional.one_hot(cluster_assignment, num_classes=self.num_clusters)
            cluster_counts = one_hot.sum(0)
            if work_dtype.is_floating_point:
                cluster_sums = one_hot.to(work_dtype).t() @ X_work
            else:
                # integer sums are exact with scatter_add_; MPS lacks integer matmul and index_add_ on torch 2.5
                index = cluster_assignment[:, None].expand(-1, X_work.shape[1])
                cluster_sums = X_work.new_zeros(self.num_clusters, X_work.shape[1]).scatter_add_(0, index, X_work)

            # edge case when a certain cluster centre has no points assigned to it:
            # just choose a random point as its update. A random index is drawn for every
            # cluster unconditionally and masked with torch.where, so the branch never depends
            # on a data-dependent Python bool (no host sync either way).
            empty_mask = (cluster_counts == 0).unsqueeze(1)
            safe_counts = cluster_counts.clamp(min=1).unsqueeze(1).to(work_dtype)
            if work_dtype.is_floating_point:
                means = cluster_sums / safe_counts
            else:
                # exact integer mean, rounded toward zero like the cast back to an integer X.dtype
                means = torch.div(cluster_sums, safe_counts, rounding_mode="trunc")
            random_points = X_work[torch.randint(len(X), (self.num_clusters,), device=X.device)]
            current_centers = torch.where(empty_mask, random_points, means).to(X.dtype)

            # sum of distance of how much the newly computed clusters have moved from their previous positions
            # (in the work dtype: subtracting and squaring in a narrow integer dtype wraps)
            shift = current_centers.to(work_dtype) - previous_centers.to(work_dtype)
            center_shift = torch.sum(torch.sqrt(torch.sum(shift**2, dim=1)))

            iteration = iteration + 1

            if self.tolerance is not None and center_shift**2 <= self.tolerance:
                break

            if self.max_iterations != 0 and iteration >= self.max_iterations:
                break

        self._final_cluster_centers = current_centers
        # The loop's assignments precede its last center update. Reassign once so labels use the returned centers.
        self._final_cluster_assignments = self.predict(X)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Find the cluster center closest to each point in x.

        Args:
            x: 2D torch.Tensor

        Returns:
            1D torch.Tensor containing cluster id assigned to each data point in x

        """
        KORNIA_CHECK_SHAPE(x, ["N", "D"])
        # x and cluster_centers should have same number of columns
        KORNIA_CHECK(
            x.shape[1] == self.cluster_centers.shape[1],
            f"Dimensions at position 1 of x and cluster_centers do not match. \
                {x.shape[1]} != {self.cluster_centers.shape[1]}",
        )

        work_dtype = self._work_dtype(x, self.cluster_centers)
        distance = self._pairwise_euclidean_distance(x.to(work_dtype), self.cluster_centers.to(work_dtype))
        return distance.argmin(-1)
