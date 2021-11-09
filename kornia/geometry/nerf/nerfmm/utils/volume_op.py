import torch


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def get_ndc_rays_fxfy(H, W, fxfy, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * fxfy[0])) * ox_oz
    o1 = -1. / (H / (2. * fxfy[1])) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * fxfy[0])) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * fxfy[1])) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def volume_sampling(c2w, ray_dir_cam, t_vals, near, far, perturb_t):
    """
    :param c2w:             (4, 4)                  camera pose
    :param ray_dir_cam:     (H, W, 3)               ray directions in the camera coordinate
    :param t_vals:          (N_sample, )            sample depth in a ray
    :param perturb_t:       True/False              whether add noise to t
    """
    ray_H, ray_W = ray_dir_cam.shape[0], ray_dir_cam.shape[1]
    N_sam = t_vals.shape[0]

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_t:
        # add some noise to each each z_val
        t_noise = torch.rand((ray_H, ray_W, N_sam), device=c2w.device, dtype=torch.float32)  # (H, W, N_sam)
        t_noise = t_noise * (far - near) / N_sam
        t_vals_noisy = t_vals.view(1, 1, N_sam) + t_noise  # (H, W, N_sam)
    else:
        t_vals_noisy = t_vals.view(1, 1, N_sam).expand(ray_H, ray_W, N_sam)

    # Get sample position in the world (1, 1, 1, 3) + (H, W, 1, 3) * (H, W, N_sam, 1) -> (H, W, N_sample, 3)
    sample_pos = ray_ori_world.view(1, 1, 1, 3) + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(3)

    return sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy  # (H, W, N_sample, 3), (3, ), (H, W, 3), (H, W, N_sam)


def volume_sampling_ndc(c2w, ray_dir_cam, t_vals, near, far, H, W, focal, perturb_t):
    """
    :param c2w:             (3/4, 4)                camera pose
    :param ray_dir_cam:     (H, W, 3)               ray directions in the camera coordinate
    :param focal:           a float or a (2,) torch tensor for focal.
    :param t_vals:          (N_sample, )            sample depth in a ray
    :param perturb_t:       True/False              whether add noise to t
    """
    ray_H, ray_W = ray_dir_cam.shape[0], ray_dir_cam.shape[1]
    N_sam = t_vals.shape[0]

    # transform rays from camera coordinate to world coordinate
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    ray_dir_world = ray_dir_world.reshape(-1, 3)  # (H, W, 3) -> (H*W, 3)
    ray_ori_world = ray_ori_world.view(1, 3).expand_as(ray_dir_world)  # (3, ) -> (1, 3) -> (H*W, 3)
    if isinstance(focal, float):
        ray_ori_world, ray_dir_world = get_ndc_rays(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    else:  # if focal is a tensor contains fxfy
        ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(H, W, focal, 1.0, rays_o=ray_ori_world,
                                                         rays_d=ray_dir_world)  # (H*W, 3)
    ray_dir_world = ray_dir_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)
    ray_ori_world = ray_ori_world.reshape(ray_H, ray_W, 3)  # (H, W, 3)

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_t:
        # add some noise to each each z_val
        t_noise = torch.rand((ray_H, ray_W, N_sam), device=c2w.device, dtype=torch.float32)  # (H, W, N_sam)
        t_noise = t_noise * (far - near) / N_sam
        t_vals_noisy = t_vals.view(1, 1, N_sam) + t_noise  # (H, W, N_sam)
    else:
        t_vals_noisy = t_vals.view(1, 1, N_sam).expand(ray_H, ray_W, N_sam)

    # Get sample position in the world (H, W, 1, 3) + (H, W, 1, 3) * (H, W, N_sam, 1) -> (H, W, N_sample, 3)
    sample_pos = ray_ori_world.unsqueeze(2) + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(3)

    return sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy  # (H, W, N_sample, 3), (3, ), (H, W, 3), (H, W, N_sam)


def volume_rendering(rgb_density, t_vals, sigma_noise_std, rgb_act_fn):
    """
    :param rgb_density:     (H, W, N_sample, 4)     network output
    :param t_vals:          (H, W, N_sample)        compute the distance between each sample points
    :param sigma_noise_std: A scalar                add some noise to the density output, this is helpful to reduce
                                                    floating artifacts according to official repo, but they set it to
                                                    zero in their implementation.
    :param rgb_act_fn:      relu()                  apply an active fn to the raw rgb output to get actual rgb
    :return:                (H, W, 3)               rendered rgb image
                            (H, W, N_sample)        weights at each sample position
    """
    ray_H, ray_W, num_sample = t_vals.shape[0], t_vals.shape[1], t_vals.shape[2]

    rgb = rgb_act_fn(rgb_density[:, :, :, :3])  # (H, W, N_sample, 3)
    sigma_a = rgb_density[:, :, :, 3]  # (H, W, N_sample)

    if sigma_noise_std > 0.0:
        sigma_noise = torch.randn_like(sigma_a) * sigma_noise_std
        sigma_a = sigma_a + sigma_noise  # (H, W, N_sample)

    sigma_a = sigma_a.relu()  # (H, W, N_sample)

    # Compute distances between samples.
    # 1. compute the distances among first (N-1) samples
    # 2. the distance between the LAST sample and infinite far is 1e10
    dists = t_vals[:, :, 1:] - t_vals[:, :, :-1]  # (H, W, N_sample-1)
    dist_far = torch.empty(size=(ray_H, ray_W, 1), dtype=torch.float32, device=dists.device).fill_(1e10)  # (H, W, 1)
    dists = torch.cat([dists, dist_far], dim=2)  # (H, W, N_sample)

    alpha = 1 - torch.exp(-1.0 * sigma_a * dists)  # (H, W, N_sample)

    # 1. We expand the exp(a+b) to exp(a) * exp(b) for the accumulated transmittance computing.
    # 2. For the space at the boundary far to camera, the alpha is constant 1.0 and the transmittance at the far boundary
    # is useless. For the space at the boundary near to camera, we manually set the transmittance to 1.0, which means
    # 100% transparent. The torch.roll() operation simply discards the transmittance at the far boundary.
    acc_transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=2)  # (H, W, N_sample)
    acc_transmittance = torch.roll(acc_transmittance, shifts=1, dims=2)  # (H, W, N_sample)
    acc_transmittance[:, :, 0] = 1.0  # (H, W, N_sample)

    weight = acc_transmittance * alpha  # (H, W, N_sample)

    # (H, W, N_sample, 1) * (H, W, N_sample, 3) = (H, W, N_sample, 3) -> (H, W, 3)
    rgb_rendered = torch.sum(weight.unsqueeze(3) * rgb, dim=2)

    depth_map = torch.sum(weight * t_vals, dim=2)  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'weight': weight,  # (H, W, N_sample)
        'depth_map': depth_map,  # (H, W)
    }
    return result
