from kornia.geometry.nerf.nerfmm.utils.pos_enc import encode_position
from kornia.geometry.nerf.nerfmm.utils.volume_op import volume_rendering, volume_sampling_ndc
from kornia.geometry.nerf.nerfmm.utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from kornia.geometry.nerf.nerfmm.utils.training_utils import mse2psnr

from kornia.geometry.nerf.nerfmm.models.intrinsics import LearnFocal
from kornia.geometry.nerf.nerfmm.models.poses import LearnPose

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple


class TinyNerf(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(TinyNerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D // 2), nn.ReLU())
        self.fc_rgb = nn.Linear(D // 2, 3)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den


class RayParameters():
    def __init__(self):
        self.NEAR, self.FAR = 0.0, 1.0  # ndc near far
        self.N_SAMPLE = 128  # samples per ray
        self.POS_ENC_FREQ = 10  # positional encoding freq for location
        self.DIR_ENC_FREQ = 4  # positional encoding freq for direction


ray_params = RayParameters()


def model_render_image(c2w, rays_cam, t_vals, ray_params, H, W, fxfy, nerf_model,
                       perturb_t, sigma_noise_std):
    """
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param perturb_t:   True/False              perturb t values.
    :param sigma_noise_std: float               add noise to raw density predictions (sigma).
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # KEY 2: sample the 3D volume using estimated poses and intrinsics online.
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, ray_params.NEAR,
                                                                     ray_params.FAR, H, W, fxfy, perturb_t)

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=ray_params.POS_ENC_FREQ, inc_input=True)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
    dir_enc = encode_position(ray_dir_world, levels=ray_params.DIR_ENC_FREQ, inc_input=True)  # (H, W, 27)
    dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, ray_params.N_SAMPLE, -1)  # (H, W, N_sample, 27)

    # inference rgb and density using position and direction encoding.
    rgb_density = nerf_model(pos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn=torch.sigmoid)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'depth_map': depth_map,  # (H, W)
    }

    return result


def train_one_epoch(imgs, H, W, ray_params, opt_nerf, opt_focal,
                    opt_pose, nerf_model, focal_net, pose_param_net):
    nerf_model.train()
    focal_net.train()
    pose_param_net.train()

    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE,
                            device='cpu')  # (N_sample,) sample position
    L2_loss_epoch = []

    # shuffle the training imgs
    N_IMGS = imgs.shape[0]

    print("N_IMGS = ", N_IMGS, " H = ", H, " W = ", W)

    ids = np.arange(N_IMGS)
    np.random.shuffle(ids)

    for i in ids:
        fxfy = focal_net()

        # KEY 1: compute ray directions using estimated intrinsics online.
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        img = imgs[i].to('cpu')  # (H, W, 4)
        c2w = pose_param_net(i)  # (4, 4)

        # sample 32x32 pixel on an image and their rays for training.
        r_id = torch.randperm(H, device='cpu')[:32]  # (N_select_rows)
        c_id = torch.randperm(W, device='cpu')[:32]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, ray_params,
                                           H, W, fxfy, nerf_model, perturb_t=True, sigma_noise_std=0.0)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        print("SHAPES: ", rgb_rendered.shape, img_selected.shape)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

        L2_loss.backward()
        opt_nerf.step()
        opt_focal.step()
        opt_pose.step()
        opt_nerf.zero_grad()
        opt_focal.zero_grad()
        opt_pose.zero_grad()

        L2_loss_epoch.append(L2_loss)

    L2_loss_epoch_mean = torch.stack(L2_loss_epoch).mean().item()
    return L2_loss_epoch_mean


def render_novel_view(c2w, H, W, fxfy, ray_params, nerf_model):
    nerf_model.eval()

    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE,
                            device='cpu')  # (N_sample,) sample position

    c2w = c2w.to('cpu')  # (4, 4)

    # split an image to rows when the input image resolution is high
    rays_dir_cam_split_rows = ray_dir_cam.split(10, dim=0)  # input 10 rows each time
    rendered_img = []
    rendered_depth = []
    for rays_dir_rows in rays_dir_cam_split_rows:
        render_result = model_render_image(c2w, rays_dir_rows, t_vals, ray_params,
                                           H, W, fxfy, nerf_model,
                                           perturb_t=False, sigma_noise_std=0.0)
        rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
        depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

        rendered_img.append(rgb_rendered_rows)
        rendered_depth.append(depth_map)

    # combine rows to an image
    rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
    rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)
    return rendered_img, rendered_depth


def train_model(imgs: torch.Tensor, n_epoch = 1)->Tuple[nn.Module, nn.Module, nn.Module]:
    # n_epoch = 1 # 200  # set to 1000 to get slightly better results. we use 10K epoch in our paper.
    EVAL_INTERVAL = 50  # render an image to visualise for every this interval.

    # Initialise all trainabled parameters
    N_IMGS = imgs.shape[0]
    H = imgs.shape[1]
    W = imgs.shape[2]
    focal_net = LearnFocal(H, W, req_grad=True)
    pose_param_net = LearnPose(num_cams=N_IMGS, learn_R=True, learn_t=True)

    # Get a tiny NeRF model. Hidden dimension set to 128
    nerf_model = TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128)

    # Set lr and scheduler: these are just stair-case exponantial decay lr schedulers.
    opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
    opt_focal = torch.optim.Adam(focal_net.parameters(), lr=0.001)
    opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)

    from torch.optim.lr_scheduler import MultiStepLR
    scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.9954)
    scheduler_focal = MultiStepLR(opt_focal, milestones=list(range(0, 10000, 100)), gamma=0.9)
    scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)

    # Set tensorboard writer
    # writer = SummaryWriter(log_dir=os.path.join('logs', scene_name, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))

    # Store poses to visualise them later
    pose_history = []

    # Training
    print('Training... Check results in the tensorboard above.')
    for epoch_i in range(n_epoch):
        L2_loss = train_one_epoch(imgs, H, W, ray_params, opt_nerf, opt_focal,
                                  opt_pose, nerf_model, focal_net, pose_param_net)
        train_psnr = mse2psnr(L2_loss)

        # writer.add_scalar('train/psnr', train_psnr, epoch_i)

        fxfy = focal_net()
        print('epoch {0:4d} Training PSNR {1:.3f}, estimated fx {2:.1f} fy {3:.1f}'.format(epoch_i, train_psnr, fxfy[0],
                                                                                           fxfy[1]))

        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()

        learned_c2ws = torch.stack([pose_param_net(i) for i in range(N_IMGS)])  # (N, 4, 4)
        pose_history.append(learned_c2ws[:, :3, 3])  # (N, 3) only store positions as we vis in 2D.

        with torch.no_grad():
            if (epoch_i + 1) % EVAL_INTERVAL == 0:
                eval_c2w = torch.eye(4, dtype=torch.float32)  # (4, 4)
                fxfy = focal_net()
                rendered_img, rendered_depth = render_novel_view(eval_c2w, H, W, fxfy, ray_params, nerf_model)
                # writer.add_image('eval/img', rendered_img.permute(2, 0, 1), global_step=epoch_i)
                # writer.add_image('eval/depth', rendered_depth.unsqueeze(0), global_step=epoch_i)

    pose_history = torch.stack(pose_history).detach().cpu().numpy()  # (n_epoch, N_img, 3)
    print('Training finished.')

    return nerf_model, focal_net, pose_param_net


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def create_spiral_poses(radii, focus_depth, n_poses=120, n_circle=2):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 2 * np.pi * n_circle, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
        # center = np.array([np.cos(t), -np.sin(t), -np.sin(t)]) * radii  # for pure zoom in

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)

