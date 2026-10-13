`ALIKED` and `DISK` no longer pad their keypoints with non-maxima, nor drop a real maximum, when an image has no more
local maxima than the keypoint budget. In top-k mode (`ALIKED(max_num_keypoints=k, detection_threshold<=0)`), ALIKED
used to return exactly `k` keypoints per image, filling the slots beyond the score map's maxima with other positions
in `torch.topk`'s tie order, including the removed border, the region a `forward_laf` `mask` excludes and the
`image_size` padding, and it raised when `k` exceeded the number of pixels. It now returns at most `k`, all of them
maxima, and `forward_laf` pads each image to the largest count of the batch with zero frames (#5862).
`DISK(...)(images, n=n)` used to drop the weakest maximum of an image with `n` maxima or fewer, and raised on an image
with none; it now returns all of them, the keypoints `n=None` returns (#5864). In threshold mode, ALIKED fell back to
the maxima above the mean score only when no image of the batch had a maximum above `detection_threshold`; the
fallback is now decided for each image, so a batch returns the keypoints that one-by-one calls return (#5865). The
default `ALIKED` configuration, and `DISK` on an image with more than `n` maxima, return the same keypoints as before.
