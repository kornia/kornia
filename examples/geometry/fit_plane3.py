# Example showing how to fit a 3d plane
import matplotlib.pyplot as plt
import torch

from kornia.geometry.liegroup import So3
from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vector3
from kornia.utils import create_meshgrid

if __name__ == "__main__":

    # define the plane
    plane_h = 25
    plane_w = 50

    # create a base mesh in the ground z == 0
    mesh = create_meshgrid(plane_h, plane_w, normalized_coordinates=True)
    X, Y = mesh[..., 0], mesh[..., 1]
    Z = 0 * X

    mesh_pts = Vector3.from_coords(X, Y, Z)

    # add noise to the mesh
    rand_pts = Vector3.random((plane_h, plane_w))
    rand_pts.z.clamp_(min=-0.1, max=0.1)

    mesh_view: Vector3 = mesh_pts + rand_pts
    # mesh_view: Vector3 = mesh_pts

    # visualize the plane as pointcloud

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(mesh_view.x, mesh_view.y, mesh_view.z, c="blue")

    # create rotation
    angle_rad = torch.tensor(3.141616 / 4)
    rot_x = So3.rot_x(angle_rad)
    rot_z = So3.rot_z(angle_rad)
    rot = rot_x * rot_z
    print(rot)

    # apply the rotation to the mesh points
    # TODO: this should work as `rot * mesh_view`
    points_rot = torch.stack([rot * x for x in mesh_view.view(-1, 3)]).detach()
    points_rot = Vector3(points_rot)

    ax.scatter(points_rot.x, points_rot.y, points_rot.z, c="green")

    # estimate the plane from the rotated points
    plane_in_ground_fit: Hyperplane = fit_plane(points_rot)
    print(plane_in_ground_fit)

    # project the original points to the estimated plane
    points_proj: Vector3 = plane_in_ground_fit.projection(mesh_view.view(-1, 3))

    ax.scatter(points_proj.x, points_proj.y, points_proj.z, c="red")
    plt.show()
