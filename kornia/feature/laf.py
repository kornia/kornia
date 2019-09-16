# flake8: noqa E127
# flake8: noqa E128

import kornia
import math
import torch
import torch.nn.functional as F


def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    """
    Returns a scale of the LAFs
    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].

    Returns:
        torch.Tensor: tensor  BxNx1x1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.get_laf_scale(input)  # BxNx1x1
    """
    n_dims = len(LAF.size())
    eps = 1e-10
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] -\
        LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps
    return out.abs().sqrt()


def make_upright(LAF: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Rectifies the affine matrix, so that it becomes upright
    Args:
        LAF: (torch.Tensor): tensor of LAFs.
        eps (float): for safe division, (default 1e-9)

    Returns:
        torch.Tensor: tensor of same shape.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.make_upright(input)  #  BxNx2x3
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    det = get_laf_scale(LAF)
    # The function is equivalent to doing 2x2 SVD and reseting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.
    b2a2 = torch.sqrt(LAF[..., 0:1, 1:2]**2 + LAF[..., 0:1, 0:1]**2) + eps
    LAF1_ell = torch.cat([(b2a2 / det).contiguous(),  # type: ignore
                          torch.zeros_like(det)], dim=3)  # type: ignore
    LAF2_ell = torch.cat([((LAF[..., 1:2, 1:2] * LAF[..., 0:1, 1:2] +
                            LAF[..., 1:2, 0:1] * LAF[..., 0:1, 0:1]) / (b2a2 * det)),
                          (det / b2a2).contiguous()], dim=3)  # type: ignore
    return torch.cat([torch.cat([LAF1_ell, LAF2_ell], dim=2),
                      LAF[..., :, 2:3]], dim=3)


def ellipse_to_laf(ells: torch.Tensor) -> torch.Tensor:
    """
    Converts ellipse regions to LAF format. Ellipse (a, b, c)
    and upright covariance matrix [a11 a12; 0 a22] are connected
    by inverse matrix square root:
    A = invsqrt([a b; b c])
    See also https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_frame2oell.m

    Args:
        ells: (torch.Tensor): tensor of ellipses in Oxford format [x y a b c].

    Returns:
        LAF: (torch.Tensor) tensor of ellipses in LAF format.

    Shape:
        - Input: :math:`(B, N, 5)`
        - Output:  :math:`(B, N, 2, 3)`
    Example:
        >>> input = torch.ones(1, 10, 5)  # BxNx5
        >>> output = kornia.ellipse_to_laf(input)  #  BxNx2x3
    """
    n_dims = len(ells.size())
    if (n_dims != 3):
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))
    B, N, dim = ells.size()
    if (dim != 5):
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))

    ell_shape = torch.cat([torch.cat([ells[..., 2:3], ells[..., 3:4]], dim=2).unsqueeze(2),
                           torch.cat([ells[..., 3:4], ells[..., 4:5]], dim=2).unsqueeze(2)],
                          dim=2).view(-1, 2, 2)
    out = torch.matrix_power(torch.cholesky(ell_shape, False), -1).view(B, N, 2, 2)
    out = torch.cat([out, ells[..., :2].view(B, N, 2, 1)], dim=3)
    return out


def laf_to_boundary_points(LAF: torch.Tensor, n_pts: int = 50) -> torch.Tensor:
    """
    Converts LAFs to boundary points of the regions + center.
    Used for local features visualization, see visualize_laf function
    Args:
        LAF: (torch.Tensor).
        n_pts: number of points to output

    Returns:
        pts: (torch.Tensor) tensor of boundary points

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [Nx2x3] or [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    B, N, _, _ = LAF.size()
    pts = torch.cat([torch.sin(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.cos(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.ones(n_pts - 1, 1)], dim=1)
    # Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0, 0, 1.]).view(1, 3),
                     pts],
                    dim=0).unsqueeze(0).expand(B * N, n_pts, 3)
    pts = pts.to(LAF.device).to(LAF.dtype)
    aux = torch.tensor([0, 0, 1.]).view(1, 1, 3).expand(B * N, 1, 3)
    HLAF = torch.cat([LAF.view(-1, 2, 3), aux.to(LAF.device).to(LAF.dtype)], dim=1)
    pts_h = torch.bmm(HLAF, pts.permute(0, 2, 1)).permute(0, 2, 1)
    return kornia.convert_points_from_homogeneous(
        pts_h.view(B, N, n_pts, 3))


def get_laf_pts_to_draw(LAF: torch.Tensor,
                        img_idx: int = 0):
    """
    Returns numpy array for drawing LAFs (local features).
    To draw:
        x, y = kornia.feature.laf.get_laf_pts_to_draw(LAF, img_idx)
        plt.figure()
        plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
        plt.plot(x, y, 'r')
        plt.show()

    Args:
        LAF: (torch.Tensor).
        n_pts: number of boundary points to output

    Returns:
        pts: (torch.Tensor) tensor of boundary points

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`
    """
    pts = laf_to_boundary_points(LAF[img_idx:img_idx + 1])[0]
    pts_np = pts.detach().permute(1, 0, 2).cpu().numpy()
    return (pts_np[..., 0], pts_np[..., 1])


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    De-normalizes LAFs from scale to image scale.
    B,N,H,W = images.size()
    MIN_SIZE = min(H,W)
    [a11 a21 x]
    [a21 a22 y]
    becomes
    [a11*MIN_SIZE a21*MIN_SIZE x*W]
    [a21*MIN_SIZE a22*MIN_SIZE y*H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [Nx2x3] or [BxNx2x3]. "
            "Got {}".format(LAF.size())
        )
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h, w)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) * min_size
    coef[0, 0, 0, 2] = w
    coef[0, 0, 1, 2] = h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    Normalizes LAFs to [0,1] scale from pixel scale.
    See below:
    B,N,H,W = images.size()
    MIN_SIZE = min(H,W)
    [a11 a21 x]
    [a21 a22 y]
    becomes
    [a11/MIN_SIZE a21/MIN_SIZE x/W]
    [a21/MIN_SIZE a22/MIN_SIZE y/H]
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size()))
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h, w)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) / min_size
    coef[0, 0, 0, 2] = 1.0 / w
    coef[0, 0, 1, 2] = 1.0 / h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img: torch.Tensor,
                                            LAF: torch.Tensor,
                                            PS: int = 32) -> torch.Tensor:
    """
    Helper function for affine grid generation.

    Args:
        img: (torch.Tensor) images, LAFs are detected in
        LAF: (torch.Tensor).
        PS (int) -- patch size to be extracted

    Returns:
        grid: (torch.Tensor).

    Shape:
        - Input: :math:`(B, CH, H, W)`,  :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, PS, PS)`
    """
    n_dims = len(LAF.size())
    if (n_dims != 4):
        raise TypeError(
            "LAF shape should be must be [BxNx2x3]. "
            "Got {}".format(LAF.size()))
    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_laf(LAF, img)

    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3),
                         [B * N, ch, PS, PS])
    grid[..., :, 0] = 2.0 * grid[..., :, 0].clone() / float(w) - 1.0
    grid[..., :, 1] = 2.0 * grid[..., :, 1].clone() / float(h) - 1.0
    return grid


def extract_patches_simple(img: torch.Tensor,
                           LAF: torch.Tensor,
                           PS: int = 32) -> torch.Tensor:
    """
    Extract patches defined by LAFs from image tensor.
    No smoothing applied, huge aliasing (better use extract_patches_from_pyramid)
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32

    Returns:
        patches: (torch.Tensor),  :math:`(B, N, CH, PS,PS)`
    """
    num, ch, h, w = img.size()
    B, N, _, _ = LAF.size()
    out = []
    # for loop temporarily, to be refactored
    for i in range(B):
        grid = generate_patch_grid_from_normalized_LAF(
            img[i:i + 1], LAF[i:i + 1], PS)
        out.append(F.grid_sample(
            img[i:i + 1].expand(grid.size(0), ch, h, w),
            grid,
            padding_mode="border")
        )
    return torch.cat(out, dim=0).view(B, N, ch, PS, PS)


def extract_patches_from_pyramid(img: torch.Tensor,
                                 LAF: torch.Tensor,
                                 PS: int = 32) -> torch.Tensor:
    """
    Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level
    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32

    Returns:
        patches: (torch.Tensor),  :math:`(B, N, CH, PS,PS)`
    """
    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(LAF, img)) / float(PS)
    pyr_idx = (scale.log2() + 0.5).relu().long()
    cur_img = img
    cur_pyr_level = int(0)
    out = torch.zeros(B, N, ch, PS, PS).to(LAF.dtype)
    while min(cur_img.size(2), cur_img.size(3)) > PS:
        num, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum()) == 0:
                continue
            scale_mask = scale_mask.byte().view(-1)
            grid = generate_patch_grid_from_normalized_LAF(
                cur_img[i:i + 1],
                LAF[i:i + 1, scale_mask, :, :],
                PS)
            out[i, scale_mask, ..., :] = out[i, scale_mask, ..., :].clone() * 0\
                + F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w),
                                grid,
                                padding_mode="border")
        cur_img = kornia.pyrdown(cur_img)
        cur_pyr_level += 1
    return out
