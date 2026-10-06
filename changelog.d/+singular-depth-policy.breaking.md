The projections of `kornia.geometry.camera` and `kornia.sensors.camera` now share one rule for a
camera-frame depth `abs(z) <= 1e-8`: the perspective divide is skipped and `K` is applied to the
undivided `(x, y)`, the rule `project_points` already followed. Elsewhere the divide is exact, with no
epsilon added to it, and a `nan` depth gives `nan`. The result at a skipped divide is a finite
placeholder, not a valid pixel; flagging it is #4555. Measured at `(1, 2, z)` with `fx = fy = 100`,
`c = (4, 3)`, before and after:

| Call | before | after |
|---|---|---|
| `PinholeCamera.project`, `z = 0` | `[100, 200]` (`K` before the skip) | `[104, 203]`, as `project_points` |
| `project_points_z1`, `Z1Projection.project`, `z = 0` | `[inf, inf]` (`nan` for a zero numerator) | `[1, 2]` |
| `project_points_z1`, `z = 5e-9` | `[2e8, 4e8]` | `[1, 2]` |
| `cam2pixel`, `z = 0` | `[1e14, 2e14]` (`inf` in float16) | `[100, 200]` |
| `cam2pixel`, `z = 1e-7` | `9.9999e8` (biased by `+ 1e-12`) | `1e9 + 4` |
| `project_points`, `PinholeCamera.project`, `z = nan` | `[104, 203]` / `nan` | `nan` |

`dx_project_points_z1` returns the Jacobian of `(x, y)`, `[[1, 0, 0], [0, 1, 0]]`, where the divide
is skipped. The pinhole and Brown-Conrady models of `kornia.sensors.camera` follow `Z1Projection`.

`cam2pixel` keeps its signature. It no longer adds `eps` to the depth: it skips the divide where
`abs(z) <= 1e-8` and divides exactly elsewhere. Its `eps` is deprecated since kornia 0.9.1 and ignored;
passing a value other than the default emits a `DeprecationWarning` in eager mode. Since `dst_proj_src`
already contains `K`, its placeholder at `z = 0` is the undivided projected point and lacks the principal
point. `DepthWarper` and `depth_warp` no longer project through `cam2pixel`: they apply the masked divide
first and `K` second, so they agree with `warp_frame_depth` where a point reaches `z = 0`, a nonzero
principal point included, instead of sampling at coordinates of order `1e12`.

`KannalaBrandtK3.project` now takes the incidence angle from the ray, `atan2(||(x, y)||, z)`: a point on
the camera plane gave `nan` and is now at 90 degrees, and a ray with `z < 0` was mirrored through the
`z = 1` plane to the opposite side of the image and now keeps its angle beyond 90 degrees
(`(0.6, 0.8, -1e-3)` gave `(-49.4, -252.5)` and gives `(690.1, 733.5)`, next to `(689.4, 732.5)` for
`z = 1e-3`). `convert_points_from_homogeneous` returns `nan` for a `nan` `w` instead of the unchanged
numerator. Fixes #4267.
