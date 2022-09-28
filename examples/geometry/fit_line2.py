# Example showing how to fit a 2d line with kornia / pytorch
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from kornia.geometry.line import ParametrizedLine, fit_line

std = 1.2  # standard deviation for the points
num_points = 50  # total number of points


if __name__ == "__main__":
    # create a baseline
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([1.0, 1.0])

    l1 = ParametrizedLine.through(p0, p1)
    print(l1)

    # sample some points and weights
    pts, w = [], []
    for t in torch.linspace(-10, 10, num_points):
        p2 = l1.point_at(t)
        p2_noise = torch.rand_like(p2) * std
        p2 += p2_noise
        pts.append(p2)
        w.append(1 - p2_noise.mean())
    pts = torch.stack(pts)
    w = torch.stack(w)

    # fit the the line
    l2 = fit_line(pts, w)
    print(l2)

    # project some points along the estimated line
    p3 = l2.point_at(-10)
    p4 = l2.point_at(10)

    # plot !

    X = torch.stack((p3, p4)).detach().numpy()
    X_pts = pts.detach().numpy()

    plt.plot(X_pts[:, 0], X_pts[:, 1], 'ro')
    plt.plot(X[:, 0], X[:, 1])
    plt.show()
