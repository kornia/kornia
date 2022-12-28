import cv2 as cv
import numpy as np
import torch
from dog_cv import homography_est_cv

from kornia.feature import match_mnn
from kornia.feature.integrated import SIFTFeature
from kornia.geometry import RANSAC
from kornia.geometry.transform import warp_perspective
from kornia.utils import image_to_tensor


class TestDog:
    def test_dog(self, device):
        img = cv.imread("imgs/boat/img1.pgm")
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_t = (image_to_tensor(img, False).float() / 255.0).to(device=device)
        Hs_gt_t = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [-1.0, 0.0, img_t.shape[-1] - 1], [0.0, 0.0, 1.0]],
                [[-1.0, 0.0, img_t.shape[-1] - 1], [0.0, -1.0, img_t.shape[-2] - 1], [0.0, 0.0, 1.0]],
                [[0.0, -1.0, img_t.shape[-2] - 1], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            device=device,
        )
        imgs_t = [img_t] + [torch.clone(torch.rot90(img_t, i, [2, 3])) for i in range(1, 4)]
        homography_est_torch(Hs_gt_t, imgs_t, device)

        Hs_gt_rot = [h.numpy() for h in Hs_gt_t]
        imgs_rot = [img] + [np.rot90(img, i, [0, 1]).copy() for i in range(1, 4)]
        homography_est_cv(Hs_gt_rot, imgs_rot)


def get_visible_part_mean_absolute_reprojection_error_torch(img1_t, img2_t, H_gt_t, H_t, device):
    """We reproject the image 1 mask to image2 and back to get the visible part mask.

    Then we average the reprojection absolute error over that area
    """
    h, w = img1_t.shape[2:]
    mask1_t = torch.ones((1, 1, h, w), device=device)

    H_gt_t = H_gt_t[None]
    mask1in2_t = warp_perspective(mask1_t, H_gt_t, img2_t.shape[2:][::-1])
    mask1inback_t = warp_perspective(mask1in2_t, torch.linalg.inv(H_gt_t), img1_t.shape[2:][::-1]) > 0

    xi_t = torch.arange(w, device=device)
    yi_t = torch.arange(h, device=device)
    xg_t, yg_t = torch.meshgrid(xi_t, yi_t, indexing='xy')

    coords_t = torch.cat(
        [xg_t.reshape(*xg_t.shape, 1), yg_t.reshape(*yg_t.shape, 1), torch.ones(*yg_t.shape, 1, device=device)], dim=2
    )

    def get_xy_rep(H_loc):
        xy_rep_t = H_loc.to(torch.float32) @ coords_t.reshape(-1, 3, 1).to(torch.float32)
        xy_rep_t /= xy_rep_t[:, 2:3]
        xy_rep_t = xy_rep_t[:, :2]
        return xy_rep_t

    xy_rep_gt_t = get_xy_rep(H_gt_t)
    xy_rep_est_t = get_xy_rep(H_t)
    error_t = torch.sqrt(((xy_rep_gt_t - xy_rep_est_t) ** 2).sum(axis=1)).reshape(xg_t.shape) * mask1inback_t[0, 0].T
    mean_error_t = error_t.sum() / mask1inback_t.sum()

    return mean_error_t.detach().cpu().item()


def homography_est_torch(Hs_gt_t, imgs_t, device):
    ransac = RANSAC(
        model_type='homography', inl_th=0.5, batch_size=2048, max_iter=100000, confidence=0.9999, max_lo_iters=5
    )

    sf = SIFTFeature(device=device)

    lafs0, responses0, descs_t0 = sf(imgs_t[0], mask=None)
    for other_i in range(1, len(imgs_t)):

        lafs1, responses1, descs_t1 = sf(imgs_t[other_i], mask=None)
        _, matches = match_mnn(descs_t0[0], descs_t1[0], dm=None)
        src_pts_t = lafs0[0, matches[:, 0], :, 2]
        dst_pts_t = lafs1[0, matches[:, 1], :, 2]

        H_est_t, inliers = ransac(src_pts_t, dst_pts_t)

        H_gt_t = Hs_gt_t[other_i - 1]
        MAE = get_visible_part_mean_absolute_reprojection_error_torch(
            imgs_t[0], imgs_t[other_i], H_gt_t, H_est_t, device
        )
        print(f"MAE torch: {MAE}")
