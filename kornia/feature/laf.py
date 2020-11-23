from typing import Union
import kornia
import math
import torch
import torch.nn.functional as F


def raise_error_if_laf_is_not_valid(laf: torch.Tensor) -> None:
    """Auxilary function, which verifies that input is a torch.tensor of [BxNx2x3] shape

    Args:
        laf
    """
    laf_message: str = "Invalid laf shape, we expect BxNx2x3. Got: {}".format(laf.shape)
    if not torch.is_tensor(laf):
        raise TypeError("Laf type is not a torch.Tensor. Got {}"
                        .format(type(laf)))
    if len(laf.shape) != 4:
        raise ValueError(laf_message)
    if laf.size(2) != 2 or laf.size(3) != 3:
        raise ValueError(laf_message)
    return


def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    """Returns a scale of the LAFs

    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].

    Returns:
        torch.Tensor: tensor  BxNx1x1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_scale(input)  # BxNx1x1
    """
    raise_error_if_laf_is_not_valid(LAF)
    eps = 1e-10
    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] - LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps
    return out.abs().sqrt()


def get_laf_center(LAF: torch.Tensor) -> torch.Tensor:
    """Returns a center (keypoint) of the LAFs

    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3].

    Returns:
        torch.Tensor: tensor  BxNx2 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 2)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_center(input)  # BxNx2
    """
    raise_error_if_laf_is_not_valid(LAF)
    out: torch.Tensor = LAF[..., 2]
    return out


def get_laf_orientation(LAF: torch.Tensor) -> torch.Tensor:
    """Returns orientation of the LAFs, in degrees.

    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3].

    Returns:
        torch.Tensor: tensor  BxNx1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_orientation(input)  # BxNx1
    """
    raise_error_if_laf_is_not_valid(LAF)
    angle_rad: torch.Tensor = torch.atan2(LAF[..., 0, 1], LAF[..., 0, 0])
    return kornia.rad2deg(angle_rad).unsqueeze(-1)


def laf_from_center_scale_ori(xy: torch.Tensor, scale: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    """Returns orientation of the LAFs, in radians. Useful to create kornia LAFs from OpenCV keypoints

    Args:
        xy: (torch.Tensor): tensor [BxNx2].
        scale: (torch.Tensor): tensor [BxNx1x1].
        ori: (torch.Tensor): tensor [BxNx1].

    Returns:
        torch.Tensor: tensor  BxNx2x3 .
    """
    names = ['xy', 'scale', 'ori']
    for var_name, var, req_shape in zip(names,
                                        [xy, scale, ori],
                                        [("B", "N", 2), ("B", "N", 1, 1), ("B", "N", 1)]):
        if not torch.is_tensor(var):
            raise TypeError("{} type is not a torch.Tensor. Got {}"
                            .format(var_name, type(var)))
        if len(var.shape) != len(req_shape):  # type: ignore  # because it does not like len(tensor.shape)
            raise TypeError(
                "{} shape should be must be [{}]. "
                "Got {}".format(var_name, str(req_shape), var.size()))
        for i, dim in enumerate(req_shape):  # type: ignore # because it wants typing for dim
            if dim is not int:
                continue
            if var.size(i) != dim:
                raise TypeError(
                    "{} shape should be must be [{}]. "
                    "Got {}".format(var_name, str(req_shape), var.size()))
    unscaled_laf: torch.Tensor = torch.cat([kornia.angle_to_rotation_matrix(ori.squeeze(-1)),
                                            xy.unsqueeze(-1)], dim=-1)
    laf: torch.Tensor = scale_laf(unscaled_laf, scale)
    return laf


def scale_laf(laf: torch.Tensor, scale_coef: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.
    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        laf: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].
        scale_coef: (torch.Tensor): broadcastable tensor or float.


    Returns:
        torch.Tensor: tensor  BxNx2x3 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Input: :math: `(B, N,)` or ()
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = scale_laf(input, scale)  # BxNx2x3
    """
    if (type(scale_coef) is not float) and (type(scale_coef) is not torch.Tensor):
        raise TypeError(
            "scale_coef should be float or torch.Tensor "
            "Got {}".format(type(scale_coef)))
    raise_error_if_laf_is_not_valid(laf)
    centerless_laf: torch.Tensor = laf[:, :, :2, :2]
    return torch.cat([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


def make_upright(laf: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Rectifies the affine matrix, so that it becomes upright

    Args:
        laf: (torch.Tensor): tensor of LAFs.
        eps (float): for safe division, (default 1e-9)

    Returns:
        torch.Tensor: tensor of same shape.

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = make_upright(input)  #  BxNx2x3
    """
    raise_error_if_laf_is_not_valid(laf)
    det = get_laf_scale(laf)
    scale = det
    # The function is equivalent to doing 2x2 SVD and reseting rotation
    # matrix to an identity: U, S, V = svd(LAF); LAF_upright = U * S.
    b2a2 = torch.sqrt(laf[..., 0:1, 1:2] ** 2 + laf[..., 0:1, 0:1] ** 2) + eps
    laf1_ell = torch.cat([(b2a2 / det).contiguous(),
                          torch.zeros_like(det)], dim=3)
    laf2_ell = torch.cat([((laf[..., 1:2, 1:2] * laf[..., 0:1, 1:2] +
                            laf[..., 1:2, 0:1] * laf[..., 0:1, 0:1]) / (b2a2 * det)),
                          (det / b2a2).contiguous()], dim=3)  # type: ignore
    laf_unit_scale = torch.cat([torch.cat([laf1_ell, laf2_ell], dim=2), laf[..., :, 2:3]], dim=3)
    return scale_laf(laf_unit_scale, scale)


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
        >>> output = ellipse_to_laf(input)  #  BxNx2x3
    """
    n_dims = len(ells.size())
    if n_dims != 3:
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))
    B, N, dim = ells.size()
    if (dim != 5):
        raise TypeError(
            "ellipse shape should be must be [BxNx5]. "
            "Got {}".format(ells.size()))
    # Previous implementation was incorrectly using Cholesky decomp as matrix sqrt
    # ell_shape = torch.cat([torch.cat([ells[..., 2:3], ells[..., 3:4]], dim=2).unsqueeze(2),
    #                       torch.cat([ells[..., 3:4], ells[..., 4:5]], dim=2).unsqueeze(2)], dim=2).view(-1, 2, 2)
    # out = torch.matrix_power(torch.cholesky(ell_shape, False), -1).view(B, N, 2, 2)

    # We will calculate 2x2 matrix square root via special case formula
    # https://en.wikipedia.org/wiki/Square_root_of_a_matrix
    # "The Cholesky factorization provides another particular example of square root
    #  which should not be confused with the unique non-negative square root."
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # M = (A 0; C D)
    # R = (sqrt(A) 0; C / (sqrt(A)+sqrt(D)) sqrt(D))
    a11 = ells[..., 2:3].abs().sqrt()
    a12 = torch.zeros_like(a11)
    a22 = ells[..., 4:5].abs().sqrt()
    a21 = ells[..., 3:4] / (a11 + a22).clamp(1e-9)
    A = torch.stack([a11, a12, a21, a22], dim=-1).view(B, N, 2, 2).inverse()
    out = torch.cat([A, ells[..., :2].view(B, N, 2, 1)], dim=3)
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
    raise_error_if_laf_is_not_valid(LAF)
    B, N, _, _ = LAF.size()
    pts = torch.cat([torch.sin(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.cos(torch.linspace(0, 2 * math.pi, n_pts - 1)).unsqueeze(-1),
                     torch.ones(n_pts - 1, 1)], dim=1)
    # Add origin to draw also the orientation
    pts = torch.cat([torch.tensor([0, 0, 1.]).view(1, 3), pts], dim=0).unsqueeze(0).expand(B * N, n_pts, 3)
    pts = pts.to(LAF.device).to(LAF.dtype)
    aux = torch.tensor([0, 0, 1.]).view(1, 1, 3).expand(B * N, 1, 3)
    HLAF = torch.cat([LAF.view(-1, 2, 3), aux.to(LAF.device).to(LAF.dtype)], dim=1)
    pts_h = torch.bmm(HLAF, pts.permute(0, 2, 1)).permute(0, 2, 1)
    return kornia.convert_points_from_homogeneous(pts_h.view(B, N, n_pts, 3))


def get_laf_pts_to_draw(LAF: torch.Tensor,
                        img_idx: int = 0):
    """Returns numpy array for drawing LAFs (local features).

    Args:
        LAF: (torch.Tensor).
        n_pts: number of boundary points to output

    Returns:
        pts: (torch.Tensor) tensor of boundary points

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, n_pts, 2)`

    Examples:
        x, y = get_laf_pts_to_draw(LAF, img_idx)
        plt.figure()
        plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
        plt.plot(x, y, 'r')
        plt.show()
    """
    # TODO: Refactor doctest
    raise_error_if_laf_is_not_valid(LAF)
    pts = laf_to_boundary_points(LAF[img_idx:img_idx + 1])[0]
    pts_np = pts.detach().permute(1, 0, 2).cpu().numpy()
    return (pts_np[..., 0], pts_np[..., 1])


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """De-normalizes LAFs from scale to image scale.
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
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    wf = float(w)
    hf = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) * min_size
    coef[0, 0, 0, 2] = wf
    coef[0, 0, 1, 2] = hf
    return coef.expand_as(LAF) * LAF


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Normalizes LAFs to [0,1] scale from pixel scale. See below:
        B,N,H,W = images.size()
        MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes:
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
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    wf: float = float(w)
    hf: float = float(h)
    min_size = min(hf, wf)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype).to(LAF.device) / min_size
    coef[0, 0, 0, 2] = 1.0 / wf
    coef[0, 0, 1, 2] = 1.0 / hf
    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img: torch.Tensor,
                                            LAF: torch.Tensor,
                                            PS: int = 32) -> torch.Tensor:
    """Helper function for affine grid generation.

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
    raise_error_if_laf_is_not_valid(LAF)
    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_laf(LAF, img)

    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3),  # type: ignore
                         [B * N, ch, PS, PS], align_corners=False)
    grid[..., :, 0] = 2.0 * grid[..., :, 0].clone() / float(w) - 1.0
    grid[..., :, 1] = 2.0 * grid[..., :, 1].clone() / float(h) - 1.0
    return grid


def extract_patches_simple(img: torch.Tensor,
                           laf: torch.Tensor,
                           PS: int = 32,
                           normalize_lafs_before_extraction: bool = True) -> torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    No smoothing applied, huge aliasing (better use extract_patches_from_pyramid)

    Args:
        img: (torch.Tensor) images, LAFs are detected in
        laf: (torch.Tensor).
        PS: (int) patch size, default = 32
        normalize_lafs_before_extraction (bool):  if True, lafs are normalized to image size, default = True

    Returns:
        patches: (torch.Tensor) :math:`(B, N, CH, PS,PS)`
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    num, ch, h, w = img.size()
    B, N, _, _ = laf.size()
    out = []
    # for loop temporarily, to be refactored
    for i in range(B):
        grid = generate_patch_grid_from_normalized_LAF(img[i:i + 1], nlaf[i:i + 1], PS).to(img.device)
        out.append(F.grid_sample(img[i:i + 1].expand(grid.size(0), ch, h, w), grid,  # type: ignore
                                 padding_mode="border", align_corners=False))
    return torch.cat(out, dim=0).view(B, N, ch, PS, PS)


def extract_patches_from_pyramid(img: torch.Tensor,
                                 laf: torch.Tensor,
                                 PS: int = 32,
                                 normalize_lafs_before_extraction: bool = True) -> torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level

    Args:
        laf: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32
        normalize_lafs_before_extraction (bool):  if True, lafs are normalized to image size, default = True

    Returns:
        patches: (torch.Tensor)  :math:`(B, N, CH, PS,PS)`
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    num, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    pyr_idx = (scale.log2() + 0.5).relu().long()
    cur_img = img
    cur_pyr_level = int(0)
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        num, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).bool().squeeze()
            if (scale_mask.float().sum()) == 0:
                continue
            scale_mask = scale_mask.bool().view(-1)
            grid = generate_patch_grid_from_normalized_LAF(
                cur_img[i:i + 1],
                nlaf[i:i + 1, scale_mask, :, :],
                PS)
            patches = F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w), grid,  # type: ignore
                                    padding_mode="border", align_corners=False)
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.pyrdown(cur_img)
        cur_pyr_level += 1
    return out


def laf_is_inside_image(laf: torch.Tensor, images: torch.Tensor,
                        border: int = 0) -> torch.Tensor:
    """Checks if the LAF is touching or partly outside the image boundary. Returns the mask
    of LAFs, which are fully inside the image, i.e. valid.

    Args:
        laf (torch.Tensor):  :math:`(B, N, 2, 3)`
        images (torch.Tensor): images, lafs are detected in :math:`(B, CH, H, W)`
        border (int): additional border

    Returns:
        mask (torch.Tensor):  :math:`(B, N)`
    """
    raise_error_if_laf_is_not_valid(laf)
    n, ch, h, w = images.size()
    pts: torch.Tensor = laf_to_boundary_points(laf, 12)
    good_lafs_mask: torch.Tensor = (pts[..., 0] >= border) *\
        (pts[..., 0] <= w - border) *\
        (pts[..., 1] >= border) *\
        (pts[..., 1] <= h - border)
    good_lafs_mask = good_lafs_mask.min(dim=2)[0]
    return good_lafs_mask


def laf_to_three_points(laf: torch.Tensor):
    """Converts local affine frame(LAF) to alternative representation: coordinates of
    LAF center, LAF-x unit vector, LAF-y unit vector.

    Args:
        laf (torch.Tensor):  :math:`(B, N, 2, 3)`

    Returns:
        threepts (torch.Tensor):  :math:`(B, N, 2, 3)`
    """
    raise_error_if_laf_is_not_valid(laf)
    three_pts: torch.Tensor = torch.stack([laf[..., 2] + laf[..., 0],
                                           laf[..., 2] + laf[..., 1],
                                           laf[..., 2]], dim=-1)
    return three_pts


def laf_from_three_points(threepts: torch.Tensor):
    """Converts three points to local affine frame.
    Order is (0,0), (0, 1), (1, 0).

    Args:
        threepts (torch.Tensor):  :math:`(B, N, 2, 3)`

    Returns:
        laf (torch.Tensor):  :math:`(B, N, 2, 3)`
    """
    laf: torch.Tensor = torch.stack([threepts[..., 0] - threepts[..., 2],
                                     threepts[..., 1] - threepts[..., 2],
                                     threepts[..., 2]], dim=-1)
    return laf
