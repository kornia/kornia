from test.geometry.nerf.test_rays import create_four_cameras

import torch

from kornia.geometry.nerf.data_utils import ImageTensors, RayDataloader, RayDataset


class TestDataset:
    def test_uniform_ray_dataset(self, device, dtype):
        cameras = create_four_cameras('cpu', torch.float32)
        imgs: ImageTensors = []
        imgs.append(torch.randint(0, 255, (3, cameras.height[0], cameras.width[0]), dtype=torch.uint8))  # (3, 5, 9)
        imgs.extend([torch.randint(0, 255, (3, 3, 4, 7), dtype=torch.uint8)[i] for i in range(3)])  # (3, 4, 7)
        dataset = RayDataset(cameras, 1, 2)
        dataset.init_ray_dataset(imgs)

        data_loader = RayDataloader(dataset, batch_size=32, shufle=False)
        d = next(iter(data_loader))  # First batch of 32 labeled rays

        # Comparing RGB values between sampled rays and original images
        assert torch.equal(d[2][0], imgs[0][:, 0, 0])
        assert torch.equal(d[2][1], imgs[0][:, 0, 1])  # First row, second column in the image (1 sample point index)
        assert torch.equal(d[2][9], imgs[0][:, 1, 0])  # Second row, first column in the image (9 sample point index)
