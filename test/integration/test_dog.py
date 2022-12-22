import math

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import kornia.utils
from kornia.feature.integrated import SIFTFeature


class SiftDescriptorWrapper:

    def __str__(self):
        return "Kornia SIFT"

    def __init__(self,):
        self.sf = SIFTFeature()

    def cv_kpt_from_laffs_responses(self, laffs, responses):
        kpts = []
        for i, response in enumerate(responses[0]):
            yx = laffs[0, i, :, 2]
            kp = cv.KeyPoint(yx[0].item(), yx[1].item(), response.item(), angle=0)
            kpts.append(kp)
        return kpts

    def detect_compute_measure(self, img, mask):

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            kpts_other, desc_other = self.detectAndCompute(img, mask)
            end.record()

            torch.cuda.synchronize()

            time = start.elapsed_time(end) / 1000
            return kpts_other, desc_other, time
        else:

            import time
            start = time.time()
            ret = self.detectAndCompute(img, mask)
            end = time.time()
            return ret + ((end - start),)

    def detectAndCompute(self, img, mask):

        if len(img.shape) == 2:
            img_np = img[:, :, None]
        else:
            img_np = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        with torch.no_grad():
            img_t3 = kornia.utils.image_to_tensor(img_np, False).float() / 255.
            img_t3 = img_t3.to(device=get_device())

            (lafs, responses, descs) = self.sf(img_t3, mask)
            kpts = self.cv_kpt_from_laffs_responses(lafs, responses)
            descs = descs[0].cpu().numpy()
            return kpts, descs


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def decompose_homographies(Hs):
    """
    :param Hs:(B, 3, 3)
    :param device:
    :return: pure_homographies(B, 3, 3), affine(B, 3, 3)
    """

    B, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

    device = get_device()
    def batched_eye_deviced(B, D):
        eye = torch.eye(D, device=device)[None].repeat(B, 1, 1)
        return eye

    KR = Hs[:, :2, :2]
    KRt = -Hs[:, :2, 2:3]
    a_t = Hs[:, 2:3, :2] @ torch.inverse(KR)
    b = a_t @ KRt + Hs[:, 2:3, 2:3]

    pure_homographies1 = torch.cat((batched_eye_deviced(B, 2), torch.zeros(B, 2, 1, device=device)), dim=2)
    pure_homographies2 = torch.cat((a_t, b), dim=2)
    pure_homographies = torch.cat((pure_homographies1, pure_homographies2), dim=1)

    affines1 = torch.cat((KR, -KRt), dim=2)
    affines2 = torch.cat((torch.zeros(B, 1, 2, device=device), torch.ones(B, 1, 1, device=device)), dim=2)
    affines = torch.cat((affines1, affines2), dim=1)

    assert torch.all(affines[:, 2, :2] == 0)
    test_compose_back = pure_homographies @ affines
    assert torch.allclose(test_compose_back, Hs, rtol=1e-01, atol=1e-01)
    return pure_homographies, affines


def get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H, metric="L2"):
    '''We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area'''
    h, w = img1.shape[:2]
    mask1 = np.ones((h, w))
    mask1in2 = cv.warpPerspective(mask1, H_gt, img2.shape[:2][::-1])
    mask1inback = cv.warpPerspective(mask1in2, np.linalg.inv(H_gt), img1.shape[:2][::-1]) > 0
    xi = np.arange(w)
    yi = np.arange(h)
    xg, yg = np.meshgrid(xi, yi)
    coords = np.concatenate([xg.reshape(*xg.shape, 1), yg.reshape(*yg.shape, 1)], axis=-1)
    xy_rep_gt = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(float), H_gt.astype(np.float32)).squeeze(1)
    xy_rep_estimated = cv.perspectiveTransform(coords.reshape(-1, 1, 2).astype(np.float32),
                                               H.astype(np.float32)).squeeze(1)
    metric = metric.upper()
    if metric == "L1":
        error = np.abs(xy_rep_gt-xy_rep_estimated).sum(axis=1).reshape(xg.shape) * mask1inback
        mean_error = error.sum() / mask1inback.sum()
    elif metric == "L2":
        error = np.sqrt(((xy_rep_gt - xy_rep_estimated) ** 2).sum(axis=1)).reshape(xg.shape) * mask1inback
        mean_error = error.sum() / mask1inback.sum()
    elif metric == "VEC":
        error = (xy_rep_estimated - xy_rep_gt).reshape(xg.shape + (2,))
        mask = np.tile(mask1inback[:, :, None], (1, 1, 2))
        error = error * mask
        mean_error = error.sum(axis=0).sum(axis=0) / mask1inback.sum()

    return mean_error


def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentative_matches, H_est, H_gt, inlier_mask, img1, img2):
    h = img1.shape[0]
    w = img1.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    def possibly_decolorize(img_local):
        if len(img_local.shape) <= 2:
            return img2
        return decolorize(img_local)

    img1_dec = possibly_decolorize(img1)
    img2_dec = possibly_decolorize(img2)

    if H_est is not None:
        dst = cv.perspectiveTransform(pts, H_est)
        img2_dec = cv.polylines(img2_dec, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
        dst = cv.perspectiveTransform(pts, H_gt)
        img2_dec = cv.polylines(img2_dec, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

    matches_mask = inlier_mask.ravel().tolist()

    # Blue is estimated homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=20)
    img_out = cv.drawMatches(img1_dec, kps1, img2_dec, kps2, tentative_matches, None, **draw_params)
    return img_out


def read_imgs(file_paths, crop=None):
    imgs = []
    for i, file in enumerate(file_paths):
        img = Image.open(file)
        img = np.array(img)

        def modulo32(n, modulo):
            n_n = n - ((n - modulo) % 32)
            assert n_n % 32 == modulo
            return n_n
        if crop:
            h, w = img.shape[:2]
            w = modulo32(w, crop)
            h = modulo32(h, crop)
            img = img[:h, :w]
        imgs.append(img)

    return imgs


def print_Hs_decomposition(Hs):

    print("scale\trotation")
    for H_gt in Hs:

        pure_homography, affine = decompose_homographies(torch.from_numpy(H_gt[None]).to(get_device()))

        affine = affine[0].cpu().numpy()

        det = np.linalg.det(affine)
        scale = math.sqrt(det)
        affine = affine / scale

        cos_avg = (affine[0, 0] + affine[1, 1]) / 2.0
        sin_avg = (affine[0, 1] - affine[1, 0]) / 2.0
        alpha = math.atan2(sin_avg, cos_avg) * 180 / math.pi
        print(f"{scale:.3f}\t{alpha:.3f}")


class TestDog:

    def test_dog(self):
        kornia_sift_descriptor = SiftDescriptorWrapper()

        Hs_bark = [
            [[0.7022029025774007, 0.4313737491020563, -127.94661199701689],
             [-0.42757325092889575, 0.6997834349758094, 201.26193857481698],
             [4.083733373964227E-6, 1.5076445750988132E-5, 1.0]],

            [[-0.48367041358997964, -0.2472935325077872, 870.2215120216712],
             [0.29085746679198893, -0.45733473891783305, 396.1604918833091],
             [-3.578663704630333E-6, 6.880007548843957E-5, 1.0]],

            [[-0.20381418476462312, 0.3510201271914591, 247.1085214229702],
             [-0.3499531830464912, -0.1975486500576974, 466.54576370699766],
             [-1.5735788289619667E-5, 1.0242951905091244E-5, 1.0]],

            [[0.30558415717792214, 0.12841186681168829, 200.94588793078017],
             [-0.12861248979242065, 0.3067557133397112, 133.77000196887894],
             [2.782320090398499E-6, 5.770764104061954E-6, 1.0]],

            [[-0.23047631546234373, -0.10655686701035443, 583.3200507850402],
             [0.11269946585180685, -0.20718914340861153, 355.2381263740649],
             [-3.580280012615393E-5, 3.2283960511548054E-5, 1.0]],
        ]

        Hs_bark = np.array(Hs_bark)
        files_bark = [f"imgs/bark/img{i + 1}.ppm" for i in range(6)]
        imgs_bark = read_imgs(files_bark)

        print("BARK experiment hompographies decomposition")
        print_Hs_decomposition(Hs_bark)
        print()

        Hs_boat = [
            [[8.5828552e-01, 2.1564369e-01, 9.9101418e+00],
             [-2.1158440e-01, 8.5876360e-01, 1.3047838e+02],
             [2.0702435e-06, 1.2886110e-06, 1.0000000e+00]],

            [[5.6887079e-01, 4.6997572e-01, 2.5515642e+01],
             [-4.6783159e-01, 5.6548769e-01, 3.4819925e+02],
             [6.4697420e-06, -1.1704138e-06, 1.0000000e+00]],

            [[1.0016637e-01, 5.2319717e-01, 2.0587932e+02],
             [-5.2345249e-01, 8.7390786e-02, 5.3454522e+02],
             [9.4931475e-06, -9.8296917e-06, 1.0000000e+00]],

            [[4.2310823e-01, -6.0670438e-02, 2.6635003e+02],
             [6.2730152e-02, 4.1652096e-01, 1.7460201e+02],
             [1.5812849e-05, -1.4368783e-05, 1.0000000e+00]],

            [[2.9992872e-01, 2.2821975e-01, 2.2930182e+02],
             [-2.3832758e-01, 2.4564042e-01, 3.6767399e+02],
             [9.9064973e-05, -5.8498673e-05, 1.0000000e+00]]
        ]
        Hs_boat = np.array(Hs_boat)
        files_boat = [f"imgs/boat/img{i + 1}.pgm" for i in range(6)]
        imgs_boat = read_imgs(files_boat)

        print("BOAT experiment hompographies decomposition")
        print_Hs_decomposition(Hs_boat)

        Hs_gt_rot, imgs_rot = Hs_imgs_for_rotation(files_bark[0])

        scales = [scale_int / 10 for scale_int in range(2, 10)]
        Hs_gt_sc_lanczos, imgs_sc_lanczos = Hs_imgs_for_scaling(files_bark[0], scales, crop_h2=True)

        run_exp(kornia_sift_descriptor, Hs_bark, imgs_bark, "bark")
        run_exp(kornia_sift_descriptor, Hs_boat, imgs_boat, "boat")
        run_exp(kornia_sift_descriptor, Hs_gt_rot, imgs_rot, "synthetic pi rotation")
        run_exp(kornia_sift_descriptor, Hs_gt_sc_lanczos, imgs_sc_lanczos, "synthetic rescaling lanczos")

        print("Results")
        print(Output.text)


def rotate(img, sin_a, cos_a, rotation_index):
    h, w = img.shape[:2]

    H_gt = np.array([
        [cos_a, sin_a, 0.],
        [-sin_a, cos_a, 0.],
        [0., 0., 1.],
    ])

    box = np.array([[0., 0., 1.], [0., h - 1, 1.], [w - 1, 0., 1.], [w - 1, h - 1, 1.]])
    box2 = (H_gt @ box.T).T
    min_x = box2[:, 0].min()
    min_y = box2[:, 1].min()

    H_gt[0, 2] = -min_x
    H_gt[1, 2] = -min_y

    bb = (w, h) if rotation_index == 2 else (h, w)
    img_rot_h = cv.warpPerspective(img, H_gt, bb)
    img_rot_r = np.rot90(img, rotation_index, [0, 1]).copy()

    assert np.all(img_rot_h == img_rot_r)
    return H_gt, img_rot_h


def gcd_euclid(a, b):

    c = a % b
    if c == 0:
        return b
    else:
        return gcd_euclid(b, c)


def get_integer_scale(scale, h, w):

    gcd = gcd_euclid(w, h)

    real_scale_gcd = round(gcd * scale)
    real_scale = real_scale_gcd / gcd

    if real_scale == 0.0 or math.fabs(real_scale - scale) > 0.1:
        raise Exception("scale {} cannot be effectively realized for w, h = {}, {} in integer domain".format(scale, w, h))

    return real_scale


def scale_img(img, scale):

    h, w = img.shape[:2]
    scale_o = scale
    scale = get_integer_scale(scale, h, w)
    print(f"scale: {scale_o} => {scale}")

    H_gt = np.array([
        [scale, 0., 0.5 * (scale - 1)],
        [0., scale, 0.5 * (scale - 1)],
        [0., 0., 1.],
    ])

    dsize = (round(w * scale), round(h * scale))
    pil = Image.fromarray(img)
    pil_resized = pil.resize(size=dsize, resample=Image.Resampling.LANCZOS)
    img_scaled = np.array(pil_resized)
    return H_gt, img_scaled


def Hs_imgs_for_rotation(file, crop=None):

    img = Image.open(file)
    img = np.array(img)

    def modulo32(n, modulo):
        n_n = n - ((n - modulo) % 32)
        assert n_n % 32 == modulo
        return n_n

    if crop:
        h, w = img.shape[:2]
        w = modulo32(w, crop)
        h = modulo32(h, crop)
        img = img[:h, :w]

    cos_a = [0., -1., 0.]
    sin_a = [1., 0., -1.]

    rotations = 3
    Hs_gt_img = [rotate(img, sin_a[i], cos_a[i], i + 1) for i in range(rotations)]
    Hs_gt = [h[0] for h in Hs_gt_img]
    imgs = [img] + [h[1] for h in Hs_gt_img]
    return Hs_gt, imgs


def Hs_imgs_for_scaling(file, scales, crop_h2=False):

    img = Image.open(file)
    img = np.array(img)
    # this assures a large gcd(w, h) and thus can be scaled without change to the aspect ratio
    if crop_h2:
        img = img[:img.shape[0] - 2]

    h_i_tuples = [scale_img(img, scale) for scale in scales]
    Hs_gt = [e[0] for e in h_i_tuples]
    imgs_r = [e[1] for e in h_i_tuples]
    imgs = [img] + imgs_r
    return Hs_gt, imgs


class Output:
    text = ""


def split_points(tentative_matches, kps0, kps1):
    src_pts = np.float32([kps0[m.queryIdx].pt for m in tentative_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps1[m.trainIdx].pt for m in tentative_matches]).reshape(-1, 2)
    kps0 = [kps0[m.queryIdx] for m in tentative_matches]
    kps1 = [kps1[m.trainIdx] for m in tentative_matches]
    return src_pts, dst_pts, kps0, kps1


def get_tentatives(kpts0, desc0, kpts1, desc1, ratio_threshold, space_dist_th=None):
    matcher = cv.BFMatcher(crossCheck=False)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)
    matches2 = matcher.match(desc1, desc0)

    tentative_matches = []
    for m, n in knn_matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue

        if space_dist_th:
            x = kpts0[m.queryIdx].pt
            y = kpts1[m.trainIdx].pt
            dist = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
            if dist > space_dist_th:
                continue

        if m.distance < ratio_threshold * n.distance:
            tentative_matches.append(m)

    src, dst, kpts0, kpts1 = split_points(tentative_matches, kpts0, kpts1)
    return src, dst, kpts0, kpts1, tentative_matches


def run_exp(descriptor, Hs_gt, imgs, e_name):

    print(f"running experiment: {e_name}")

    metric_names = ["MAE", "running time", "tentatives", "inliers"]

    data = [[] for _ in enumerate(metric_names)]

    ratio_threshold = 0.8

    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    metrics = []

    print("original")
    kpts_0, desc_0, time_0 = descriptor.detect_compute_measure(imgs[0], mask=None)

    for other_i in range(1, len(imgs)):
        print(f"query img no. {other_i}")

        kpts_other, desc_other, time_other = descriptor.detect_compute_measure(imgs[other_i], mask=None)

        time = time_0 + time_other
        src_pts, dst_pts, _, _, tentative_matches = get_tentatives(kpts_0, desc_0, kpts_other, desc_other, ratio_threshold)
        if len(src_pts) < 4:
            print(f"WARNING: less than 4 tentatives: {len(src_pts)}")
            na = "N/A"
            metrics.append([na, time, na, na])
            continue
        H_est, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,
                                               maxIters=ransac_iters, ransacReprojThreshold=ransac_th,
                                               confidence=ransac_conf)

        H_gt = Hs_gt[other_i - 1]
        MAE = get_visible_part_mean_absolute_reprojection_error(imgs[0], imgs[other_i], H_gt, H_est, metric="L2")

        tent_count = len(src_pts)
        in_count = inlier_mask.sum()
        metrics.append([MAE, time, tent_count, in_count])

    for i_m, metric_name in enumerate(metric_names):
        metric_info = []
        metric_info.append(metric_name)
        sum = 0
        for i in range(len(metrics)):
            val = metrics[i][i_m]
            if not val:
                continue
            metric_info.append(val)
            if type(val) != str:
                sum += val
        metric_info.append(sum)
        prepend = []
        if i_m == 0:
            prepend = [str(descriptor)]
        data[i_m].append(prepend + metric_info)

    def format_data(d):
        s = ""
        for i in range(len(d[0])):
            s += "\t".join([str(d[j][i]) for j in range(len(d))]) + "\n"
        return s
    Output.text += f"\n\n experiment: {e_name}\n\n"
    for d in data:
        Output.text += f"\n{format_data(d)}"


if __name__ == "__main__":
    TestDog().test_dog()
