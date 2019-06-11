from typing import Optional

import os
import torch


def save_pointcloud_ply(filename: str, pointcloud: torch.Tensor) -> None:
    r"""Utility function to save to disk a pointcloud in PLY format.

    Args:
        filename (str): the path to save the pointcloud.
        pointcloud (torch.Tensor): tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.
    """
    if not isinstance(filename, str) and filename[-3:] == '.ply':
        raise TypeError("Input filename must be a string in with the .ply  "
                        "extension. Got {}".format(filename))
    if not torch.is_tensor(pointcloud):
        raise TypeError("Input pointcloud type is not a torch.Tensor. Got {}"
                        .format(type(pointcloud)))
    if not len(pointcloud.shape) == 3 and pointcloud.shape[-1] == 3:
        raise TypeError("Input pointcloud must be in the following shape "
                        "HxWx3. Got {}.".format(pointcloud.shape))
    # flatten the input pointcloud in a vector to iterate points
    xyz_vec: torch.Tensor = pointcloud.reshape(-1, 3)

    with open(filename, 'w') as f:
        data_str: str = ''
        num_points: int = xyz_vec.shape[0]
        for idx in range(num_points):
            xyz = xyz_vec[idx]
            if not bool(torch.isfinite(xyz).any()):
                continue
            x: float = xyz[0].item()
            y: float = xyz[1].item()
            z: float = xyz[2].item()
            data_str += '{0} {1} {2}\n'.format(x, y, z)

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment arraiy generated\n")
        f.write("element vertex %d\n" % num_points)
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")
        f.write(data_str)


def load_pointcloud_ply(
        filename: str, header_size: Optional[int] = 8) -> torch.Tensor:
    r"""Utility function to load from disk a pointcloud in PLY format.

    Args:
        filename (str): the path to the pointcloud.
        header_size (Optional[int]): the size of the ply file header that will
          be skipped during loading. Default is 8 lines.

    Return:
        torch.Tensor: a tensor containing the loaded point with shape
          :math:`(*, 3)` where :math:`*` represents the number of points.
    """
    if not isinstance(filename, str) and filename[-3:] == '.ply':
        raise TypeError("Input filename must be a string in with the .ply  "
                        "extension. Got {}".format(filename))
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")
    if not (isinstance(header_size, int) and header_size > 0):
        raise TypeError("Input header_size must be a positive integer. Got {}."
                        .format(header_size))
    # open the file and populate tensor
    with open(filename, 'r') as f:
        points = []

        # skip header
        lines = f.readlines()[header_size:]

        # iterate over the points
        for line in lines:
            x_str, y_str, z_str = line.split()
            points.append((
                torch.tensor(float(x_str)),
                torch.tensor(float(y_str)),
                torch.tensor(float(z_str)),
            ))

        # create tensor from list
        pointcloud: torch.Tensor = torch.tensor(points)
        return pointcloud
