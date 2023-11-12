# reference: Pyramidal Implementation of the Lucas Kanade Feature Tracker
# url: http://robots.stanford.edu/cs223b04/algo_tracking.pdf
from __future__ import annotations

import torch

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE
from kornia.filters.filter import filter2d
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.transform.pyramid import build_pyramid

__all__ = ["optical_flow_lk"]


@torch.no_grad()
def optical_flow_lk(
    prev_image: Tensor,
    next_image: Tensor,
    prev_points: Tensor,
    window_size: int | tuple[int, int] = 3,
    max_level: int = 3,
    offsets=None,
    max_count: int = 30,
    threshold: float = 0.03,
    compile_onnx: bool = False,
) -> Tensor:
    KORNIA_CHECK_SHAPE(prev_image, ["C", "H", "W"])
    KORNIA_CHECK(prev_image.shape == next_image.shape, "prev_image and next_image must have the same shape.")
    KORNIA_CHECK_SHAPE(prev_points, ["N", "2"])
    KORNIA_CHECK(max_level >= 1, "max_level must be greater than 1.")

    N, _ = prev_points.shape

    kernel_x = torch.tensor([[[-1, 1], [-1, 1]]], device=prev_image.device, dtype=prev_image.dtype)
    kernel_y = torch.tensor([[[-1, -1], [1, 1]]], device=prev_image.device, dtype=prev_image.dtype)

    prev_image_pyramid = build_pyramid(prev_image[None], max_level=max_level)
    next_image_pyramid = build_pyramid(next_image[None], max_level=max_level)

    if offsets is None:
        w = window_size // 2
        offset_x = torch.linspace(-w, w, window_size, device=prev_image.device, dtype=prev_image.dtype)
        offset_y = torch.linspace(-w, w, window_size, device=prev_image.device, dtype=prev_image.dtype)
        offset_y, offset_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
        offsets = torch.stack([offset_x, offset_y], dim=-1)[None]  # 1xWxWx2

    v_guess = torch.zeros(N, 2, device=prev_image.device, dtype=prev_image.dtype)

    i = max_level - 1  # start from the bottom of the pyramid

    while i >= 0:
        # take the current pyramid images
        prev_image_i = prev_image_pyramid[i]
        next_image_i = next_image_pyramid[i]

        height_i, width_i = prev_image_i.shape[-2:]

        # compute the gradients dx, dy, dt
        Ix_i = filter2d(prev_image_i, kernel_x, border_type="replicate")[0]
        Iy_i = filter2d(prev_image_i, kernel_y, border_type="replicate")[0]

        # scale the offsets to the image size
        prev_points_i = prev_points / (2**i)

        prev_points_i_norm = normalize_pixel_coordinates(prev_points_i, height_i, width_i)  # Nx2

        prev_image_points_i = torch.nn.functional.grid_sample(
            prev_image_i, prev_points_i_norm[None, None, ...], align_corners=True
        )  # 1x1x1xN

        # compute the grid to sample the gradients
        prev_points_i_offsets = prev_points_i[:, None, None, :] + offsets  # Nx2
        prev_points_i_offsets_norm = normalize_pixel_coordinates(prev_points_i_offsets, height_i, width_i)  # NxWxWx2

        # make the grid to sample the gradients
        prev_points_i_offsets_norm = prev_points_i_offsets_norm.reshape(1, N, -1, 2).repeat(2, 1, 1, 1)  # 2xNx(W*W)x2

        # sample the gradients to form the system of equations
        Ixy_i_points = torch.nn.functional.grid_sample(
            torch.stack([Ix_i, Iy_i], dim=0), prev_points_i_offsets_norm, align_corners=True
        )  # 2x1xNx(W*W)

        # make the system of equations
        Ix = Ixy_i_points[0, 0]  # Nx(WxW)
        Iy = Ixy_i_points[1, 0]

        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        G_i = torch.stack(
            [
                Ix2.sum(dim=-1),
                Ixy.sum(dim=-1),
                Ixy.sum(dim=-1),
                Iy2.sum(dim=-1),
            ],
            dim=-1,
        ).reshape(N, 2, 2)

        v = torch.zeros(N, 2, device=prev_image.device, dtype=prev_image.dtype)

        # NOTE: this is not ideal since we cannot break the loop

        results = []

        for k in range(max_count):
            prev_points_k = prev_points_i + v_guess + v  # Nx2
            prev_points_k_norm = normalize_pixel_coordinates(prev_points_k, height_i, width_i)  # Nx2

            next_image_points_k = torch.nn.functional.grid_sample(
                next_image_i, prev_points_k_norm[None, None, :, :], align_corners=True
            )  # 1x1xNx2

            Ik = (prev_image_points_i - next_image_points_k)[0, 0, 0]  # 4

            bk = torch.stack(
                [
                    (Ik[:, None] * Ix).sum(dim=-1),
                    (Ik[:, None] * Iy).sum(dim=-1),
                ],
                dim=-1,
            )  # Nx2

            # compute the inverse of G_j
            if compile_onnx:
                det_G_i = G_i[:, 0, 0] * G_i[:, 1, 1] - G_i[:, 0, 1] * G_i[:, 1, 0]
                G_i_inv = (
                    torch.stack(
                        [
                            G_i[:, 1, 1],
                            -G_i[:, 0, 1],
                            -G_i[:, 1, 0],
                            G_i[:, 0, 0],
                        ],
                        dim=-1,
                    ).reshape(N, 2, 2)
                    / det_G_i[:, None, None]
                )
                nk = G_i_inv @ bk
            else:
                # NOTE: this is not supported by ONNX
                nk = torch.linalg.lstsq(G_i, bk).solution

            v += nk  # 2

            results.append(nk.norm())

            # TODO: mask the points that converged
            if bool(nk.norm() < threshold):
                results.append(k)
                break

        print(f"LK: {i}")
        print(results)

        # update for next level
        v_guess = 2 * (v + v_guess)

        # go one level up
        i -= 1

    return prev_points + v_guess
