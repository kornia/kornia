from test.geometry.nerf.test_rays import create_four_cameras

import torch

from kornia.geometry.camera import PinholeCamera
from kornia.geometry.nerf.data_utils import ImageTensors, RayDataset, instantiate_ray_dataloader
from kornia.testing import assert_close


def create_random_images_for_cameras(cameras: PinholeCamera) -> ImageTensors:
    torch.manual_seed(112)
    imgs: ImageTensors = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        imgs.append(torch.randint(0, 255, (3, int(height), int(width)), dtype=torch.uint8))  # (C, H, W)
    return imgs


def create_red_images_for_cameras(cameras: PinholeCamera) -> ImageTensors:
    imgs: ImageTensors = []
    for height, width in zip(cameras.height.tolist(), cameras.width.tolist()):
        red_img = torch.zeros(3, int(height), int(width), dtype=torch.uint8)  # (C, H, W)
        red_img[0, ...] = 255
        imgs.append(red_img)
    return imgs


class TestDataset:
    def test_uniform_ray_dataset(self, device, dtype):
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        dataset = RayDataset(cameras, 1, 2, device=device)
        dataset.init_ray_dataset()
        dataset.init_images_for_training(imgs)

        batch_size = 32
        data_loader = instantiate_ray_dataloader(dataset, batch_size=batch_size, shufle=False)
        d = next(iter(data_loader))  # First batch of 32 labeled rays

        # Check dimensions
        assert d[0].shape == (batch_size, 3)  # Ray origins
        assert d[1].shape == (batch_size, 3)  # Ray directions
        assert d[2].shape == (batch_size, 3)  # Ray rgbs

        # Comparing RGB values between sampled rays and original images
        assert_close(d[2][0].cpu(), imgs[0][:, 0, 0] / 255.0)
        assert_close(
            d[2][1].cpu(), imgs[0][:, 0, 1] / 255.0
        )  # First row, second column in the image (1 sample point index)
        assert_close(
            d[2][9].cpu(), imgs[0][:, 1, 0] / 255.0
        )  # Second row, first column in the image (9 sample point index)
