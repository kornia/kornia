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
