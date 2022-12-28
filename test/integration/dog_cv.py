import numpy as np
import torch
import cv2 as cv
from kornia.feature.integrated import SIFTFeature
from kornia.utils import image_to_tensor


def cv_kpt_from_laffs_responses(laffs, responses):
    kpts = []
    for i, response in enumerate(responses[0]):
        yx = laffs[0, i, :, 2]
        kp = cv.KeyPoint(yx[0].item(), yx[1].item(), response.item(), angle=0)
        kpts.append(kp)
    return kpts


def detect_and_compute(sift_feature, img, mask):
    img_t = image_to_tensor(img, False).float() / 255.0
    (lafs, responses, descs) = sift_feature(img_t, mask)
    kpts = cv_kpt_from_laffs_responses(lafs, responses)
    descs = descs[0].detach().cpu().numpy()
    return kpts, descs


def split_points(tentative_matches, kps0, kps1):
    src_pts = np.float32([kps0[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps1[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    kps0 = [kps0[m.queryIdx] for m in tentative_matches]
    kps1 = [kps1[m.trainIdx] for m in tentative_matches]
    return src_pts, dst_pts, kps0, kps1


def get_tentatives(kpts0, desc0, kpts1, desc1, ratio_threshold):
    matcher = cv.BFMatcher(crossCheck=False)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    matches2 = matcher.match(desc1, desc0)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue

        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    src, dst, kpts0, kpts1 = split_points(tentative_matches, kpts0, kpts1)
    return src, dst, kpts0, kpts1, tentative_matches


def get_visible_part_mean_absolute_reprojection_error_np(img1, img2, H_gt, H):
    """We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area
    """

    h, w = img1.shape[:2]
    mask1 = np.ones((h, w))
    mask1in2 = cv.warpPerspective(mask1, H_gt, img2.shape[:2][::-1])
    mask1inback = cv.warpPerspective(mask1in2, np.linalg.inv(H_gt), img1.shape[:2][::-1]) > 0
    xi = np.arange(w)
    yi = np.arange(h)
    xg, yg = np.meshgrid(xi, yi)

    coords = np.concatenate([xg.reshape(*xg.shape, 1), yg.reshape(*yg.shape, 1)], axis=-1)

    xy_rep_gt = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(float), H_gt.astype(np.float32)).squeeze(1)
    xy_rep_estimated = cv.perspectiveTransform(
        coords.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32)
    ).squeeze(1)

    error = np.sqrt(((xy_rep_gt - xy_rep_estimated) ** 2).sum(axis=1)).reshape(xg.shape) * mask1inback
    mean_error = error.sum() / mask1inback.sum()
    return mean_error


def homography_est_cv(Hs_gt, imgs):

    sift_feature = SIFTFeature(device=torch.device("cpu"))

    ratio_threshold = 0.8
    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    kpts_0, desc_0 = detect_and_compute(sift_feature, imgs[0], mask=None)

    for other_i in range(1, len(imgs)):

        kpts_other, desc_other = detect_and_compute(sift_feature, imgs[other_i], mask=None)
        src_pts, dst_pts, _, _, tentative_matches = get_tentatives(
            kpts_0, desc_0, kpts_other, desc_other, ratio_threshold
        )
        H_est, inlier_mask = cv.findHomography(
            src_pts, dst_pts, cv.RANSAC, maxIters=ransac_iters, ransacReprojThreshold=ransac_th, confidence=ransac_conf
        )

        H_gt = Hs_gt[other_i - 1]
        MAE = get_visible_part_mean_absolute_reprojection_error_np(imgs[0], imgs[other_i], H_gt, H_est)
        print(f"MAE cv: {MAE}")
        assert MAE < 0.01
