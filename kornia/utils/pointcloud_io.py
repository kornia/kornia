import os

import pypose as pp
import torch


def save_pointcloud_ply(filename: str, pointcloud: torch.Tensor) -> None:
    r"""Utility function to save to disk a pointcloud in PLY format.

    Args:
        filename: the path to save the pointcloud.
        pointcloud: tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.
    """
    if not isinstance(filename, str) and filename[-3:] == ".ply":
        raise TypeError(f"Input filename must be a string in with the .ply  extension. Got {filename}")

    if not torch.is_tensor(pointcloud):
        raise TypeError(f"Input pointcloud type is not a torch.Tensor. Got {type(pointcloud)}")

    if not len(pointcloud.shape) >= 2 and pointcloud.shape[-1] == 3:
        raise TypeError(f"Input pointcloud must be in the following shape HxWx3. Got {pointcloud.shape}.")

    # flatten the input pointcloud in a vector to iterate points
    xyz_vec: torch.Tensor = pointcloud.reshape(-1, 3)

    with open(filename, "w") as f:
        data_str: str = ""
        num_points: int = xyz_vec.shape[0]
        for idx in range(num_points):
            xyz = xyz_vec[idx]
            if not bool(torch.isfinite(xyz).any()):
                num_points -= 1
                continue
            x: float = float(xyz[0])
            y: float = float(xyz[1])
            z: float = float(xyz[2])
            data_str += f"{x} {y} {z}\n"

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment arraiy generated\n")
        f.write("element vertex %d\n" % num_points)
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")
        f.write(data_str)


def load_pointcloud_ply(filename: str, header_size: int = 8) -> torch.Tensor:
    r"""Utility function to load from disk a pointcloud in PLY format.

    Args:
        filename: the path to the pointcloud.
        header_size: the size of the ply file header that will
          be skipped during loading.

    Return:
        tensor containing the loaded point with shape :math:`(*, 3)` where
        :math:`*` represents the number of points.
    """
    if not isinstance(filename, str) and filename[-3:] == ".ply":
        raise TypeError(f"Input filename must be a string in with the .ply  extension. Got {filename}")
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError(f"Input header_size must be a positive integer. Got {header_size}.")
    # open the file and populate tensor
    with open(filename) as f:
        points = []

        # skip header
        lines = f.readlines()[header_size:]

        # iterate over the points
        for line in lines:
            x_str, y_str, z_str = line.split()
            points.append((torch.tensor(float(x_str)), torch.tensor(float(y_str)), torch.tensor(float(z_str))))

        # create tensor from list
        pointcloud: torch.Tensor = torch.tensor(points)
        return pointcloud


def iterative_closest_point(
    points_in_a: torch.Tensor, points_in_b: torch.Tensor, max_iterations: int = 20, tolerance: float = 1e-4
) -> torch.Tensor:
    """Compute the relative transformation between two point clouds.

    The resulting transformation uses the iterative closest point algorithm to satisfy:

        points_in_b = b_from_a @ points_in_a

    Args:
        points_in_a: The point cloud in the source coordinates frame A  with shape Nx3
        points_in_b:  The point cloud in the source coordinates frame A with shape Nx
        max_iterations (int): Maximum number of iterations to run.
        tolerance (float): Tolerance criteria for stopping.

     Return:
        The relative transformation between the two pointcloud with shape 3x4
    """

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

    return pp.svdtf(src, points_in_a)
