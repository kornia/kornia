import torch


def comp_ray_dir_cam(H, W, focal):
    """Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: (H, W, 3) torch.float32
    """
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W) / focal  # (H, W)
    dirs_y = -(y - 0.5*H) / focal  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    return rays_dir


def comp_ray_dir_cam_fxfy(H, W, fx, fy):
    """Compute ray directions in the camera coordinate, which only depends on intrinsics.
    This could be further transformed to world coordinate later, using camera poses.
    :return: (H, W, 3) torch.float32
    """
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=fx.device),
                          torch.arange(W, dtype=torch.float32, device=fx.device))  # (H, W)

    # Use OpenGL coordinate in 3D:
    #   x points to right
    #   y points to up
    #   z points to backward
    #
    # The coordinate of the top left corner of an image should be (-0.5W, 0.5H, -1.0).
    dirs_x = (x - 0.5*W) / fx  # (H, W)
    dirs_y = -(y - 0.5*H) / fy  # (H, W)
    dirs_z = -torch.ones(H, W, dtype=torch.float32, device=fx.device)  # (H, W)
    rays_dir = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    return rays_dir