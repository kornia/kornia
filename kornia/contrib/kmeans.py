# based on https://github.com/subhadarship/kmeans_pytorch

import torch


class KMeans:
    def __init__(
        self,
        num_clusters: int,
        cluster_centers: torch.Tensor,
        tolerance: float = 10e-4,
        max_iterations: int = 0,
        device: torch.device = torch.device('cpu'),
        seed: int = None,
    ) -> None:
        if num_clusters == 0:
            raise ValueError("num_clusters can't be 0")

        self.num_clusters = num_clusters
        self.cluster_centers = cluster_centers
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.device = device

        self.final_cluster_assignments = None
        self.final_cluster_centers = None

        if seed is not None:
            torch.manual_seed(seed)

    def get_cluster_centers(self) -> torch.Tensor:
        if self.final_cluster_centers is None:
            raise ValueError("Model has not been fit to a dataset")
        return self.final_cluster_centers.cpu()

    def get_cluster_assignments(self) -> torch.Tensor:
        if self.final_cluster_assignments is None:
            raise ValueError("Model has not been fit to a dataset")
        return self.final_cluster_assignments.cpu()

    def _initialise_cluster_centers(self, X: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Chooses num_cluster points from X as the initial cluster centers.

        Args:
            X: 2D input tensor to be clustered
            num_clusters: number of desired cluster centers
        Return:
            2D Tensor with num_cluster rows
        """
        num_samples = len(X)
        perm = torch.randperm(num_samples)
        idx = perm[:num_clusters]
        initial_state = X[idx]
        return initial_state

    def _pairwise_euclidean_distance(self, data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """Computes pairwise distance between 2 sets of vectors.

        Args:
            data1: 2D tensor of shape N, D
            data2: 2D tensor of shape C, D
        Return:
            2D tensor of shape N, C
        """
        # N*1*D
        A = data1.unsqueeze(dim=1)
        # 1*C*D
        B = data2.unsqueeze(dim=0)
        distance = (A - B) ** 2.0
        # return N*C matrix for pairwise distance
        distance = distance.sum(dim=-1)
        return distance

    def fit(self, X: torch.Tensor) -> None:
        """Iterative KMeans clustering till a threshold for shift in cluster centers
        or a maximum no of iterations have reached
        Args:
            X: 2D input tensor to be clustered
        """
        # X should have only 2 dimensions
        if not len(X.shape) == 2:
            raise ValueError(f"Invalid shape for X, we expect N x D. Got: {X.shape}")

        if self.cluster_centers is None:
            self.cluster_centers = self._initialise_cluster_centers(X, self.num_clusters)
        else:
            # cluster_centers should have only 2 dimensions
            if not len(self.cluster_centers.shape) == 2:
                raise ValueError(f"Invalid cluster_centers shape, we expect C x D. Got: {self.cluster_centers.shape}")

            # X and cluster_centers should have same number of columns
            if not X.shape[1] == self.cluster_centers.shape[1]:
                raise ValueError(
                    f"Dimensions at position 1 of X and cluster_centers do not match. \
                        {X.shape[1]} != {self.cluster_centers.shape[1]}"
                )

        X = X.float()
        X = X.to(self.device)
        current_centers = self.cluster_centers.to(self.device)

        previous_centers = None
        iteration = 0

        while True:
            # find distance between X and current_centers
            distance = self._pairwise_euclidean_distance(X, current_centers)

            cluster_assignment = torch.argmin(distance, dim=1)

            previous_centers = current_centers.clone()

            for index in range(self.num_clusters):
                selected = torch.nonzero(cluster_assignment == index).squeeze().to(self.device)
                selected = torch.index_select(X, 0, selected)
                # edge case when a certain cluster centre has no points assigned to it
                # just choose a random point as it's update
                if selected.shape[0] == 0:
                    selected = X[torch.randint(len(X), (1,))]
                current_centers[index] = selected.mean(dim=0)

            # sum of distance of how much the newly computed clusters have moved from their previous positions
            center_shift = torch.sum(torch.sqrt(torch.sum((current_centers - previous_centers) ** 2, dim=1)))

            iteration = iteration + 1

            if self.tolerance is not None and center_shift**2 < self.tolerance:
                break

            if self.max_iterations != 0 and iteration >= self.max_iterations:
                break

        self.final_cluster_assignments = cluster_assignment
        self.final_cluster_centers = current_centers

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Find the cluster center closest to each point in x
        Args:
            x: 2D tensor
        Return:
            1D tensor containing cluster id assigned to each data point in x
        """

        # x and cluster_centers should have same number of columns
        if not x.shape[1] == self.final_cluster_centers.shape[1]:
            raise ValueError(
                f"Dimensions at position 1 of x and cluster_centers do not match.\
                    {x.shape[1]} != {self.final_cluster_centers.shape[1]}"
            )

        x = x.float()
        x = x.to(self.device)
        distance = self._pairwise_euclidean_distance(x, self.final_cluster_centers)
        cluster_assignment = torch.argmin(distance, axis=1)
        return cluster_assignment.cpu()


# create example dataset
# torch.manual_seed(2023)
# x = 5 * torch.randn((500, 2)) + torch.tensor((-13, 17))
# x = torch.vstack([x, torch.randn((500, 2)) + torch.tensor((15, -12))])
# x = torch.vstack([x, 13 * torch.randn((500, 2)) + torch.tensor((35, 15))])

# kmeans = KMeans(3, None, 10e-4, 10000, torch.device('cuda'), 2023)
# kmeans.fit(x)

# # assignments = kmeans.get_cluster_assignments()
# centers = kmeans.get_cluster_centers()
# prediciton = kmeans.predict(torch.tensor([[2, 3], [5, 6]]))

# import matplotlib.pyplot as plt

# plt.scatter(x[:,0], x[:,1], c='red')
# plt.scatter(centers[:,0], centers[:,1], c='blue')
# plt.show()

# print(centers, prediciton)
