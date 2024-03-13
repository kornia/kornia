import pytest
import torch

import kornia

from testing.base import BaseTester


class TestKMeans(BaseTester):
    @pytest.mark.parametrize("num_clusters", [3, 10, 1])
    @pytest.mark.parametrize("tolerance", [10e-4, 10e-5, 10e-1])
    @pytest.mark.parametrize("max_iterations", [10, 1000])
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
    @pytest.mark.parametrize("tolerance", [10e-4])
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
        with pytest.raises(Exception) as errinfo:
            kornia.contrib.KMeans(0, None, 10e-4, 10, 0)
        assert "num_clusters can't be 0" in str(errinfo)

        # case: cluster centers is not a 2D tensor
        with pytest.raises(TypeError) as errinfo:
            starting_centers = torch.rand((2, 3, 5), device=device, dtype=dtype)
            kmeans = kornia.contrib.KMeans(None, starting_centers, 10e-4, 100, 0)
        assert "shape must be [['C', 'D']]. Got torch.Size([2, 3, 5])" in str(errinfo)

        # case: input data is not a 2D tensor
        with pytest.raises(TypeError) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, 0)
            kmeans.fit(torch.rand((1000, 5, 60), dtype=dtype, device=device))
        assert "shape must be [['N', 'D']]. Got torch.Size([1000, 5, 60])" in str(errinfo)

        # case: column dimensions of cluster centers and data to be predicted do not match
        with pytest.raises(Exception) as errinfo:
            kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, 0)
            kmeans.fit(torch.rand((1000, 5), dtype=dtype))
            kmeans.predict(torch.rand((10, 7), dtype=dtype))
        assert "7 != 5" in str(errinfo)

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

        kmeans = kornia.contrib.KMeans(3, None, 10e-4, 10000, 2023)
        kmeans.fit(x)

        centers = kmeans.cluster_centers
        prediciton = kmeans.predict(torch.tensor([[-14, 16], [45, 12]], dtype=dtype, device=device)).tolist()

        expected_centers = torch.tensor([[-13, 17], [15, -12], [35, 15]], dtype=dtype, device=device)
        expected_prediciton = [0, 2]

        # sorting centers using dimension 0 as key so that they can be checked for equalness
        order = torch.argsort(centers[:, 0]).tolist()
        new_classes = {old_class: new_class for new_class, old_class in enumerate(order)}

        ordered_centers = centers[order]
        oredered_prediciton = [new_classes[predicted_class] for predicted_class in prediciton]

        self.assert_close(ordered_centers, expected_centers, atol=2, rtol=0.1)
        assert oredered_prediciton == expected_prediciton

    def test_dynamo(self, device, dtype, torch_optimizer):
        x = TestKMeans._create_data(device, dtype)
        kmeans_params = (3, None, 10e-4, 10000, 2023)
        predict_param = torch.tensor([[-14, 16], [45, 12]], dtype=dtype, device=device)

        kmeans = kornia.contrib.KMeans(*kmeans_params)
        kmeans.fit(x)

        centers = kmeans.cluster_centers
        prediciton = kmeans.predict(predict_param)

        kmeans_op = kornia.contrib.KMeans(*kmeans_params)
        kmeans_op.fit = torch_optimizer(kmeans_op.fit)
        kmeans_op.predict = torch_optimizer(kmeans_op.predict)

        kmeans_op.fit(x)

        centers_op = kmeans_op.cluster_centers
        prediciton_op = kmeans_op.predict(predict_param)

        self.assert_close(centers, centers_op)
        self.assert_close(prediciton, prediciton_op)
