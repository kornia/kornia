import os
import unittest
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from kornia.geometry.nerf.from_nerfmm import render_novel_view, TinyNerf, train_model, train_one_epoch
from kornia.geometry.nerf.nerfmm.models.intrinsics import LearnFocal
from kornia.geometry.nerf.nerfmm.models.poses import LearnPose
from kornia.geometry.nerf.nerfmm.utils.training_utils import mse2psnr
from kornia.utils import image_list_to_tensor


class RayParameters:
    def __init__(self):
        self.NEAR, self.FAR = 0.0, 1.0  # ndc near far
        self.N_SAMPLE = 128  # samples per ray
        self.POS_ENC_FREQ = 10  # positional encoding freq for location
        self.DIR_ENC_FREQ = 4  # positional encoding freq for direction


class CameraCalibration:
    def __init__(self,
                 nerf_model: nn.Module = TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128),
                 device: str = 'cpu'):
        self._nerf_model: nn.Module = nerf_model
        self._focal_net: nn.Module = None
        self._pose_param_net: nn.Module = None

        self._ray_params: RayParameters = None

        self._opt_nerf: torch.optim.Optimizer = None
        self._opt_focal: torch.optim.Optimizer = None
        self._opt_pose: torch.optim.Optimizer = None

        self._scheduler_nerf: MultiStepLR = None
        self._scheduler_focal: MultiStepLR = None
        self._scheduler_pose: MultiStepLR = None

        self._device = device

        self._images: torch.Tensor = None
        self._image_filenames: List[str] = None

    def load_images(self, path, num_of_images_to_use: int = None):
        images: List[np.array] = []
        self._image_filenames: List[str] = []
        for i, filename in enumerate(os.listdir(path)):
            if num_of_images_to_use is not None and i >= num_of_images_to_use:
                break
            img_bgr: np.array = cv2.imread(os.path.join(path, filename))
            img_bgr = img_bgr.astype(np.float32) / 255
            images.append(img_bgr)

            self._image_filenames.append(filename)
        self._images = image_list_to_tensor(images)

        # FIXME: This part needs to be removed. It is here because the convension in NERFMM is (B, H, W, C), and in Kornia it is (B, C, H, W)
        self._images = self._images.permute(0, 2, 3, 1)

    def get_image_filenames(self, camera_id: int) -> str:
        return self._image_filenames[camera_id]

    def set_images(self, images: torch.Tensor):
        self._images = images

    def get_image_sizes(self) -> Tuple[int, int, int, int]:
        return self._images.shape[0], self._images.shape[1], self._images.shape[2], self._images.shape[3]

    def init_training(self):
        n_imgs = self._images.shape[0]
        h = self._images.shape[1]
        w = self._images.shape[2]
        self._focal_net = LearnFocal(h, w, req_grad=True)
        self._pose_param_net = LearnPose(num_cams=n_imgs, learn_R=True, learn_t=True)
        self._ray_params = RayParameters()

        if self._device == 'cuda':
            self._focal_net.cuda()
            self._pose_param_net.cuda()
            self._nerf_model.cuda()

        self._opt_nerf: torch.optim.Optimizer = torch.optim.Adam(self._nerf_model.parameters(), lr=0.001)
        self._opt_focal: torch.optim.Optimizer = torch.optim.Adam(self._focal_net.parameters(), lr=0.001)
        self._opt_pose: torch.optim.Optimizer = torch.optim.Adam(self._pose_param_net.parameters(), lr=0.001)

        self._scheduler_nerf = MultiStepLR(self._opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.9954)
        self._scheduler_focal = MultiStepLR(self._opt_focal, milestones=list(range(0, 10000, 100)), gamma=0.9)
        self._scheduler_pose = MultiStepLR(self._opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)

    def run(self, n_epoch: int = 1):
        h = self._images.shape[1]
        w = self._images.shape[2]

        # Training
        for epoch_i in range(n_epoch):
            l2_loss = train_one_epoch(self._images, h, w, self._ray_params, self._opt_nerf, self._opt_focal,
                                      self._opt_pose, self._nerf_model, self._focal_net, self._pose_param_net,
                                      self._device)
            train_psnr = mse2psnr(l2_loss)

            fxfy = self._focal_net()
            print('epoch {:4d} Training PSNR {:.3f}, estimated fx {:.1f} fy {:.1f}'.format(epoch_i, train_psnr, fxfy[0],
                                                                                           fxfy[1]))

            self._scheduler_nerf.step()
            self._scheduler_focal.step()
            self._scheduler_pose.step()

    def model(self) -> nn.Module:
        return self._nerf_model

    def focals(self) -> Tuple[float, float]:
        fxfy = self._focal_net()
        return fxfy[0], fxfy[1]

    def pose(self, camera_id: int):
        return self._pose_param_net(camera_id)

    def n_epoch(self) -> int:
        return self._n_epoch

    def render_image_from_training_set(self, camera_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fxfy = self._focal_net()
        h = self._images.shape[1]
        w = self._images.shape[2]
        c2w = self._pose_param_net(camera_id)
        with torch.no_grad():
            rendered_img, rendered_depth = render_novel_view(c2w, h, w, fxfy, self._ray_params, self._nerf_model,
                                                             self._device)
        rendered_img = rendered_img[:, :, [2, 1, 0]]
        return rendered_img, rendered_depth


class MyTestCase(unittest.TestCase):
    def test_nerf(self):
        x = CameraCalibration()
        images = torch.rand(2, 40, 50, 3)
        x.add_images(images)
        x.init_training()
        x.run(n_epoch=2)

        img, depth = x.render_image_from_training_set(1)
        print(img.shape, depth.shape)


if __name__ == '__main__':
    unittest.main()
