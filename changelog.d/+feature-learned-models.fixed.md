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
