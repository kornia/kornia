from test.geometry.nerf.test_rays import create_four_cameras

import torch

from kornia.geometry.nerf.data_utils import ImageTensors, RayDataloader, RayDataset


class TestDataset:
    def test_uniform_ray_dataset(self, device, dtype):
        cameras = create_four_cameras('cpu', torch.float32)
        imgs: ImageTensors = []
        imgs.append(torch.randint(0, 255, (3, cameras.height[0], cameras.width[0]), dtype=torch.uint8))
        imgs.extend([torch.randint(0, 255, (3, 3, 4, 7), dtype=torch.uint8)[i] for i in range(3)])
        dataset = RayDataset(cameras, 1, 2)
        dataset.init_ray_dataset(imgs)

        data_loader = RayDataloader(dataset, batch_size=32)

        # data_loader = DataLoader(
        #     dataset, sampler=BatchSampler(RandomSampler(dataset), 32, drop_last=False), collate_fn=collate_rays
        # )

        d = next(iter(data_loader))
        print(d)
