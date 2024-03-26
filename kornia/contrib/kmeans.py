# based on https://github.com/subhadarship/kmeans_pytorch

from __future__ import annotations

import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.geometry.linalg import euclidean_distance


class KMeans:
    """Implements the kmeans clustering algorithm with euclidean distance as similarity measure.

    Args:
        num_clusters: number of clusters the data has to be assigned to
        cluster_centers: tensor of starting cluster centres can be passed instead of num_clusters
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
        cluster_centers: Tensor | None,
        tolerance: float = 10e-4,
        max_iterations: int = 0,
        seed: int | None = None,
    ) -> None:
        KORNIA_CHECK(num_clusters != 0, "num_clusters can't be 0")

        # cluster_centers should have only 2 dimensions
        if cluster_centers is not None:
            KORNIA_CHECK_SHAPE(cluster_centers, ["C", "D"])

        self.num_clusters = num_clusters
        self._cluster_centers = cluster_centers
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self._final_cluster_assignments: None | Tensor = None
        self._final_cluster_centers: None | Tensor = None

        if seed is not None:
            torch.manual_seed(seed)

    @property
    def cluster_centers(self) -> Tensor:
        if isinstance(self._final_cluster_centers, Tensor):
            return self._final_cluster_centers
        if isinstance(self._cluster_centers, Tensor):
            return self._cluster_centers
        else:
            raise TypeError("Model has not been fit to a dataset")

    @property
    def cluster_assignments(self) -> Tensor:
        if isinstance(self._final_cluster_assignments, Tensor):
            return self._final_cluster_assignments
        else:
            raise TypeError("Model has not been fit to a dataset")

    def _initialise_cluster_centers(self, X: Tensor, num_clusters: int) -> Tensor:
        """Chooses num_cluster points from X as the initial cluster centers.

        Args:
            X: 2D input tensor to be clustered
            num_clusters: number of desired cluster centers

        Returns:
            2D Tensor with num_cluster rows
        """
        num_samples: int = len(X)
        perm = torch.randperm(num_samples, device=X.device)
        idx = perm[:num_clusters]
        initial_state = X[idx]
        return initial_state

    def _pairwise_euclidean_distance(self, data1: Tensor, data2: Tensor) -> Tensor:
        """Computes pairwise squared distance between 2 sets of vectors.

        Args:
            data1: 2D tensor of shape N, D
            data2: 2D tensor of shape C, D

        Returns:
            2D tensor of shape N, C
        """
        # N*1*D
        A = data1[:, None, ...]
        # 1*C*D
        B = data2[None, ...]
        distance = euclidean_distance(A, B)
        return distance

    def fit(self, X: Tensor) -> None:
        """Iterative KMeans clustering till a threshold for shift in cluster centers or a maximum no of iterations
        have reached.

        Args:
            X: 2D input tensor to be clustered
        """
        # X should have only 2 dimensions
        KORNIA_CHECK_SHAPE(X, ["N", "D"])

        if self._cluster_centers is None:
            self._cluster_centers = self._initialise_cluster_centers(X, self.num_clusters)
        else:
            # X and cluster_centers should have same number of columns
            KORNIA_CHECK(
                X.shape[1] == self._cluster_centers.shape[1],
                f"Dimensions at position 1 of X and cluster_centers do not match. \
                {X.shape[1]} != {self._cluster_centers.shape[1]}",
            )

        # X = X.to(self.device)
        current_centers = self._cluster_centers

        previous_centers: Tensor | None = None
        iteration: int = 0

        while True:
            # find distance between X and current_centers
            distance: Tensor = self._pairwise_euclidean_distance(X, current_centers)

            cluster_assignment = distance.argmin(-1)

            previous_centers = current_centers.clone()

            for index in range(self.num_clusters):
                selected = torch.nonzero(cluster_assignment == index).squeeze()
                selected = torch.index_select(X, 0, selected)
                # edge case when a certain cluster centre has no points assigned to it
                # just choose a random point as it's update
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,), device=X.device)]
                current_centers[index] = selected.mean(dim=0)

            # sum of distance of how much the newly computed clusters have moved from their previous positions
            center_shift = torch.sum(torch.sqrt(torch.sum((current_centers - previous_centers) ** 2, dim=1)))

            iteration = iteration + 1

            if self.tolerance is not None and center_shift**2 < self.tolerance:
                break

            if self.max_iterations != 0 and iteration >= self.max_iterations:
                break

        self._final_cluster_assignments = cluster_assignment
        self._final_cluster_centers = current_centers

    def predict(self, x: Tensor) -> Tensor:
        """Find the cluster center closest to each point in x.

        Args:
            x: 2D tensor

        Returns:
            1D tensor containing cluster id assigned to each data point in x
        """

        # x and cluster_centers should have same number of columns
        KORNIA_CHECK(
            x.shape[1] == self.cluster_centers.shape[1],
            f"Dimensions at position 1 of x and cluster_centers do not match. \
                {x.shape[1]} != {self.cluster_centers.shape[1]}",
        )

        distance = self._pairwise_euclidean_distance(x, self.cluster_centers)
        cluster_assignment = distance.argmin(-1)
        return cluster_assignment
