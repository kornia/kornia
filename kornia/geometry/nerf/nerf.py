import os
import unittest
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from kornia.geometry.nerf.from_nerfmm import render_novel_view, TinyNerf, train_one_epoch
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


class LearnFocals(nn.Module):
    def __init__(self, h: int, w: int, req_grad: bool):
        super().__init__()
        self.fxfy = nn.Parameter(torch.ones(size=(num_cams, 2), dtype=torch.float32), requires_grad=req_grad)

    def forward(self, cam_id):
        pass


class CameraCalibration:
    def __init__(self,
                 nerf_model: nn.Module = TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128),
                 device: str = 'cpu', checkpoint_path: str = None, checkpoint_save_frequency: int = 1,
                 n_selected_rays: int = 32):
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

        self._checkpoint_path = checkpoint_path
        self._checkpoint_save_frequency = checkpoint_save_frequency

        self._n_selected_rays = n_selected_rays

        self._images: torch.Tensor = None
        self._image_filenames: List[str] = None

        self._epoch0 = 0

    def load_images(self, path, num_of_images_to_use: int = None):
        images: List[np.array] = []
        self._image_filenames: List[str] = []
        for i, filename in enumerate(os.listdir(path)):
            if num_of_images_to_use is not None and i >= num_of_images_to_use:
                break
            img_bgr: np.array = cv2.imread(os.path.join(path, filename))
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

    def __init_optimizers(self):
        self._opt_nerf: torch.optim.Optimizer = torch.optim.Adam(self._nerf_model.parameters(), lr=0.001)
        self._opt_focal: torch.optim.Optimizer = torch.optim.Adam(self._focal_net.parameters(), lr=0.001)
        self._opt_pose: torch.optim.Optimizer = torch.optim.Adam(self._pose_param_net.parameters(), lr=0.001)

    def init_training(self, load_checkpoint_path=None):
        n_imgs = self._images.shape[0]
        h = self._images.shape[1]
        w = self._images.shape[2]

        self._epoch0 = 0
        self._focal_net = LearnFocal(h, w, req_grad=True)
        self._pose_param_net = LearnPose(num_cams=n_imgs, learn_R=True, learn_t=True)
        if load_checkpoint_path is not None:
            self.__load_model_checkpoint(load_checkpoint_path)

        else:
            self.__to_device()
            self.__init_optimizers()

        # FIXME: Verify how these schedulers work when models and optimizers are loaded from a checkpoint
        self._scheduler_nerf = MultiStepLR(self._opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.9954)
        self._scheduler_focal = MultiStepLR(self._opt_focal, milestones=list(range(0, 10000, 100)), gamma=0.9)
        self._scheduler_pose = MultiStepLR(self._opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)

        self._ray_params = RayParameters()

    def run(self, n_epoch: int = 1):
        h = self._images.shape[1]
        w = self._images.shape[2]

        # Training
        for epoch_i in range(self._epoch0, self._epoch0 + n_epoch):

            # fxfy = self._focal_net()
            # print('focals before epoch: {:0.1f}, {:0.1f}'.format(fxfy[0], fxfy[1]))

            l2_loss = train_one_epoch(self._images, h, w, self._ray_params, self._n_selected_rays,
                                      self._opt_nerf, self._opt_focal, self._opt_pose,
                                      self._nerf_model, self._focal_net, self._pose_param_net,
                                      self._device)
            train_psnr = mse2psnr(l2_loss)

            fxfy = self._focal_net()
            print('epoch {:4d} Training PSNR {:.3f}, estimated fx {:.1f} fy {:.1f}'.format(epoch_i + 1, train_psnr, fxfy[0],
                                                                                           fxfy[1]))

            self._scheduler_nerf.step()
            self._scheduler_focal.step()
            self._scheduler_pose.step()

            if self._checkpoint_path is not None and (epoch_i + 1) % self._checkpoint_save_frequency == 0:
                self.__save_model_checkpoint(epoch_i + 1, train_psnr)

    def __save_model_checkpoint(self, epoch: int, train_psnr: float):
        print('Saving model checkpoint')
        torch.save({
            'epoch': epoch,
            'nerf_model_state_dict': self._nerf_model.state_dict(),
            'focal_net_state_dict': self._focal_net.state_dict(),
            'pose_param_net_state_dict': self._pose_param_net.state_dict(),
            'nerf_optimizer_state_dict': self._opt_nerf.state_dict(),
            'focal_optimizer_state_dict': self._opt_focal.state_dict(),
            'pose_optimizer_state_dict': self._opt_pose.state_dict(),
            'train_psnr': train_psnr}, self._checkpoint_path)

    def __to_device(self):
        device = torch.device(self._device)
        self._nerf_model.to(device)
        self._focal_net.to(device)
        self._pose_param_net.to(device)

    def __load_model_checkpoint(self, load_checkpoint_path: str):
        device = torch.device(self._device)
        checkpoint = torch.load(load_checkpoint_path)
        self._nerf_model.load_state_dict(checkpoint['nerf_model_state_dict'])
        self._focal_net.load_state_dict(checkpoint['focal_net_state_dict'])
        self._pose_param_net.load_state_dict(checkpoint['pose_param_net_state_dict'])
        self.__to_device()
        self.__init_optimizers()
        self._opt_nerf.load_state_dict(checkpoint['nerf_optimizer_state_dict'])
        self._opt_focal.load_state_dict(checkpoint['focal_optimizer_state_dict'])
        self._opt_pose.load_state_dict(checkpoint['pose_optimizer_state_dict'])
        self._epoch0 = checkpoint['epoch']

    def model(self) -> nn.Module:
        return self._nerf_model

    def focals(self) -> Tuple[float, float]:
        fxfy = self._focal_net()
        return fxfy[0], fxfy[1]

    def pose(self, camera_id: int):
        return self._pose_param_net(camera_id)

    def camera_matrix(self, camera_id: int) -> torch.Tensor:
        device = torch.device('cpu')
        rt = self.pose(camera_id).to(device)
        rt = rt[0:3]
        fxfy = self._focal_net().to(device)
        h = self._images.shape[1]
        w = self._images.shape[2]
        k = torch.tensor([[fxfy[0], 0,       h / 2],
                          [0,       fxfy[1], w / 2],
                          [0,       0,       1]])
        return torch.matmul(k, rt)

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
        x.set_images(images)
        x.init_training()
        x.run(n_epoch=2)

        img, depth = x.render_image_from_training_set(1)
        print(img.shape, depth.shape)


if __name__ == '__main__':
    unittest.main()
