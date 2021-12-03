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
    def __init__(self, h: int, w: int, num_cams: int, req_grad: bool, multi_focal: bool):
        super().__init__()
        self.__h = h
        self.__w = w
        self.__multi_focal = multi_focal
        if multi_focal:
            self.__fxfy = \
                nn.Parameter(torch.tensor(torch.ones(size=(num_cams, 2)), dtype=torch.float32), requires_grad=req_grad)
        else:
            self.__fxfy = \
                nn.Parameter(torch.tensor(torch.ones(size=(1, 2)), dtype=torch.float32), requires_grad=req_grad)

    def forward(self, cam_id):
        if self.__multi_focal:
            fx = self.__fxfy[cam_id][0]
            fy = self.__fxfy[cam_id][1]
        else:
            fx = self.__fxfy[0][0]
            fy = self.__fxfy[0][1]
        fxfy = torch.stack([fx**2 * self.__w, fy**2 * self.__h])
        return fxfy

    def is_multi_focal(self):
        return self.__multi_focal


class CameraCalibration:
    def __init__(self,
                 nerf_model: nn.Module = TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128),
                 device: str = 'cpu', checkpoint_path: str = None, checkpoint_save_frequency: int = 1,
                 n_selected_rays: int = 32,
                 lr: float = 0.001):
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

        self._lr = lr

        self._multi_focal: bool = None

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

    def __init_camera_models(self, multi_focal: bool):
        n_imgs = self._images.shape[0]
        h = self._images.shape[1]
        w = self._images.shape[2]
        self._focal_net = LearnFocals(h, w, n_imgs, req_grad=True, multi_focal=multi_focal)
        self._pose_param_net = LearnPose(num_cams=n_imgs, learn_R=True, learn_t=True)

    def __init_optimizers(self):     # FIXME: handle learning rate parameterization properly
        self._opt_nerf: torch.optim.Optimizer = torch.optim.Adam(self._nerf_model.parameters(), lr=self._lr)
        self._opt_focal: torch.optim.Optimizer = torch.optim.Adam(self._focal_net.parameters(), lr=self._lr)
        self._opt_pose: torch.optim.Optimizer = torch.optim.Adam(self._pose_param_net.parameters(), lr=self._lr)

    def __reset_nerf(self):
        for layer in self._nerf_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def init_training(self, load_checkpoint_path=None, reset_only_nerf=False, multi_focal=True):
        if reset_only_nerf and (not self._focal_net or not self._pose_param_net):
            raise RuntimeError('Cannot reset only Nerf model parameters if camera models were not instantiated')
        self._epoch0 = 0

        if load_checkpoint_path is not None:
            self.__load_model_checkpoint(load_checkpoint_path)

        else:
            if not reset_only_nerf:
                self.__init_camera_models(multi_focal)
                self._multi_focal = multi_focal
            self.__reset_nerf()
            self.__to_device()
            self.__init_optimizers()

        if reset_only_nerf:
            self.__reset_nerf()

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

            fxfy = self._focal_net(0)
            if not self._focal_net.is_multi_focal():
                print('epoch {:4d} Training PSNR {:.3f}, estimated fx {:.1f} fy {:.1f}'
                      .format(epoch_i + 1, train_psnr, fxfy[0], fxfy[1]))
            else:
                fxfy2 = self._focal_net(1)
                print('epoch {:4d} Training PSNR {:.3f}, estimated fx {:.1f} fy {:.1f}; fx {:.1f} fy {:.1f}'
                      .format(epoch_i + 1, train_psnr, fxfy[0], fxfy[1], fxfy2[0], fxfy2[1]))

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
            'multi_focal': self._multi_focal,
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
        checkpoint = torch.load(load_checkpoint_path)
        self._nerf_model.load_state_dict(checkpoint['nerf_model_state_dict'])
        self._multi_focal = checkpoint['multi_focal']
        self.__init_camera_models(self._multi_focal)
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

    def focals(self, camera_id: int) -> Tuple[float, float]:
        fxfy = self._focal_net(camera_id)
        return fxfy[0], fxfy[1]

    def pose(self, camera_id: int):
        return self._pose_param_net(camera_id)

    def camera_matrix(self, camera_id: int) -> torch.Tensor:
        device = torch.device('cpu')
        rt = self.pose(camera_id).to(device)
        rt = rt[0:3]
        fxfy = self._focal_net(camera_id).to(device)
        h = self._images.shape[1]
        w = self._images.shape[2]
        k = torch.tensor([[fxfy[0], 0,       h / 2],
                          [0,       fxfy[1], w / 2],
                          [0,       0,       1]])
        return torch.matmul(k, rt)

    def n_epoch(self) -> int:
        return self._n_epoch

    def render_image_from_training_set(self, camera_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fxfy = self._focal_net(camera_id)
        h = self._images.shape[1]
        w = self._images.shape[2]
        c2w = self._pose_param_net(camera_id)
        with torch.no_grad():
            rendered_img, rendered_depth = render_novel_view(c2w, h, w, fxfy, self._ray_params, self._nerf_model,
                                                             self._device)
        rendered_img = rendered_img[:, :, [2, 1, 0]]
        return rendered_img, rendered_depth


def initialize_nerf_object_for_testing(checkpoint_path: str = None) -> CameraCalibration:
    x = CameraCalibration(checkpoint_path = checkpoint_path)
    images = torch.rand(3, 40, 50, 3)
    x.set_images(images)
    return x

def equal_models(model1: nn.Module, model2: nn.Module) -> bool:
    models_equal = True
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            models_equal = False
            break
    return models_equal

import copy


class MyTestCase(unittest.TestCase):
    def test_nerf(self):
        x = CameraCalibration()
        images = torch.rand(3, 40, 50, 3)
        x.set_images(images)
        x.init_training()
        x.run(n_epoch=5)

        img, depth = x.render_image_from_training_set(1)
        print(img.shape, depth.shape)

    def test_initialize_and_run(self):
        x = initialize_nerf_object_for_testing()
        x.init_training()
        x.run(n_epoch=2)

    def test_initialize_from_checkpoint(self):
        checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')
        x = initialize_nerf_object_for_testing(checkpoint_path=checkpoint_path)
        x.init_training()
        x.run(n_epoch=2)
        fxfy = x.focals(0)
        pose = x.pose(0)

        x = initialize_nerf_object_for_testing()
        x.init_training(load_checkpoint_path=checkpoint_path)
        fxfy_after_load = x.focals(0)
        self.assertEqual(fxfy, fxfy_after_load)
        pose_after_load = x.pose(0)
        self.assertTrue(torch.Tensor.equal(pose, pose_after_load))

    def test_initialize_only_nerf_and_run(self):
        x = initialize_nerf_object_for_testing()
        with self.assertRaises(RuntimeError):
            x.init_training(reset_only_nerf=True)
        x.init_training()
        x.run(n_epoch=2)
        fxfy = x.focals(0)
        pose = x.pose(0)
        model = copy.deepcopy(x.model())

        x.init_training(reset_only_nerf=True)

        fxfy_after_reset = x.focals(0)
        self.assertEqual(fxfy, fxfy_after_reset)
        pose_after_reset = x.pose(0)
        self.assertTrue(torch.Tensor.equal(pose, pose_after_reset))
        model_after_reset = x.model()

        self.assertFalse(equal_models(model, model_after_reset))

    def test_initialize_run_and_initialize(self):
        x = initialize_nerf_object_for_testing()
        x.init_training()
        x.run(n_epoch=2)
        model = copy.deepcopy(x.model())

        x.init_training()
        model_after_init = x.model()

        self.assertFalse(equal_models(model, model_after_init))


if __name__ == '__main__':
    unittest.main()
    # MyTestCase().test_initialize_run_and_initialize()
