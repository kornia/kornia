# Example showing how to fit a 2d line with kornia / pytorch
import matplotlib.pyplot as plt

from kornia.geometry.plane import Hyperplane, fit_plane
from kornia.geometry.vector import Vec3
from kornia.utils import create_meshgrid

std = 1.2  # standard deviation for the points
num_points = 4  # total number of points


if __name__ == "__main__":

    # generate a batch of random three-d points
    plane_h = 50
    plane_w = 50
    rand_pts = Vec3.random([plane_h, plane_w])

    # define points from
    p0 = Vec3.from_coords(0.0, 0.0, 0.0)
    p1 = Vec3.from_coords(0.0, 1.0, 0.0)
    p2 = Vec3.from_coords(0.0, 0.0, 1.0)

    mesh = create_meshgrid(plane_h, plane_w, False)[0]
    X, Y = mesh.permute(2, 0, 1)
    Z = 0 * X

    mesh_pts = Vec3.from_coords_tensor(X, Y, Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect(aspect=(1, 1, 0.1))

    mesh_view = mesh_pts + rand_pts

    ax.scatter(mesh_view.x, mesh_view.y, mesh_view.z)

    # three-d plane
    plane_in_ground = Hyperplane.through(p0, p1, p2)
    print(plane_in_ground)

    mesh_projected = plane_in_ground.projection(mesh_view)

    ax.plot_surface(mesh_projected.x, mesh_projected.y, mesh_projected.z, alpha=0.5)
    plt.show()

    rand_points_projected: Vec3 = plane_in_ground.projection(rand_pts)
    plane_in_ground_fit: Hyperplane = fit_plane(rand_points_projected)
    print(plane_in_ground_fit)
    pass
