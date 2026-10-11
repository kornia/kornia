`ALIKED` now detects keypoints with the non-maximum suppression of its reference implementation (Shiaoming/ALIKED and
LightGlue's port), which keeps tied maxima and, in two further rounds, the maxima of what remains once the windows
around the maxima already kept are zeroed. Previously one strict `nms2d` pass dropped both, so `ALIKED` and
`ALIKED.forward_laf` returned a subset of the reference keypoints (1363 of the reference's 1514 on Oxford graf img1
with the pretrained `aliked-n16`); they now return the reference set, and `nms_radius` still sets the window.
`ALIKED.forward_laf` now returns upright, isotropic frames `s * I`, where `s` is the root-mean-square distance in pixels
from the keypoint to the positions its descriptor samples (about 10 to 28 px with the pretrained `aliked-n16`).
Previously the 2x2 block came from the covariance of the soft-argmax weights: a sub-pixel scale, an orientation set by
the eigenvector order and a mirrored frame (`det < 0`). `compute_affine=False` still returns identity blocks.
`LightGlueMatcher("aliked")` reads only keypoint positions and descriptors, so its matches change with the keypoint set
but not with the frames.

`DISK` and `XFeat` now keep keypoints on the first and last two rows and columns of the image, as their reference
implementations (cvlab-epfl/disk and verlab/accelerated_features) do. Both used `nms2d`, which never reports a maximum
within `window_size // 2` pixels (2 with the default window) of the border, so on Oxford graf and boat they returned
fewer keypoints than the references (`XFeat` lost 7 of 4096 on graf img1 and 56 and 62 of about 3000 on two crops;
`DISK(n=None)` lost 1 of 4410 on graf img1 and 6 of 7368 on boat img1). They now run the references' padded max-pool NMS
and return the reference keypoint sets exactly, with identical scores and descriptors. A window holding equal maxima now
keeps all of them in `XFeat` and the first of them in `DISK`, where `nms2d` kept none. `DISK` raises `ValueError` for
an even `window_size`, as the reference does. `nms2d` itself is unchanged.

`DeDoDe.detect` now pads an image to a multiple of 14 the way `DeDoDe.forward` and `DeDoDe.describe` do: after the
ImageNet normalisation, so the pad is the mean colour. Previously `detect` called directly padded first and normalised
after, so the network saw a black strip on the bottom and right, and on a 370x504 crop of Oxford graf img1 it selected
94 of 2000 keypoints different from `forward` (`describe` and `forward` already normalised first). `DeDoDe.describe` now
also cuts the padding off its descriptor map: with `keypoints=None` it returns the `(B, DIM, H, W)` map its docstring
states instead of a map padded to a multiple of 14, and the descriptors it samples at the normalised keypoints of
`detect` come from the right pixels when `H` or `W` is not a multiple of 14. `detect` followed by `describe` now returns
exactly what `forward` returns, and images whose sides are multiples of 14 are bitwise unchanged. The mean-colour pad
is the closer one to the reference implementation (Parskatt/DeDoDe), which does not pad.

`SOLD2` and `SOLD2_detector` now pick junctions like the reference implementation (cvg/SOLD2). The junction suppression
keeps the stronger of two junctions that are exactly 8 pixels apart along both axes or closer (`grid_size` in general),
where it kept both at 8 pixels; and the junctions, hence the endpoints of every line segment, are in raster order, so
the first endpoint is the one with the smaller row. Before, that held only with the default
`use_junction_refinement=True`: with `False` the endpoints followed the junction score, and `[[191, 261], [60, 39]]`
came out where the reference returns `[[60, 39], [191, 261]]`. With the pretrained weights on Oxford graf img1 the
junction set drops from 416 to the reference's 399 and the 246 segments become the reference's 229, identical to the
coordinate in both modes.
