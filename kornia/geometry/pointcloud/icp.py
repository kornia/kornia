import pypose as pp
import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


def iterative_closest_point(
    points_in_a: torch.Tensor,
    points_in_b: torch.Tensor,
    max_iterations: int = 20,
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute the relative transformation between two point clouds.

    The resulting transformation uses the iterative closest point algorithm to satisfy:

        points_in_b = b_from_a @ points_in_a

    Args:
        points_in_a: The point cloud in the source coordinates frame A  with shape Nx3
        points_in_b:  The point cloud in the source coordinates frame A with shape Nx3
        max_iterations (int): Maximum number of iterations to run.
        tolerance (float): Tolerance criteria for stopping.

     Return:
        The relative transformation between the two pointcloud with shape 3x4
    """

    KORNIA_CHECK_SHAPE(points_in_a, ["B", "3"])
    KORNIA_CHECK_SHAPE(points_in_b, ["B", "3"])

    src = points_in_a.clone()
    prev_error = 0

    for iter in range(max_iterations):
        distances = torch.cdist(src, points_in_b)
        min_idx = torch.argmin(distances, dim=1)

        a_mean = torch.mean(src, dim=0)
        b_mean = torch.mean(points_in_b[min_idx], dim=0)

        a_center = src - a_mean
        b_center = points_in_b[min_idx] - b_mean

        H = a_center.T @ b_center

        U, S, Vt = torch.linalg.svd(H)

        R = Vt.T @ U.T

        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = b_mean.T - R @ a_mean.T

        src = (R @ src.T + t.unsqueeze(-1)).T

        mean_error = torch.mean(torch.norm(src - points_in_b[min_idx], dim=1))
        if torch.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        if verbose:
            print(f"ICP Mean error: {mean_error}, iter: {iter}")

    return pp.svdtf(src, points_in_a)