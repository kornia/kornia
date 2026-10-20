`equalize`, `equalize3d`, `RandomEqualize` and `RandomEqualize3D`, and `equalize_clahe` and `RandomClahe` on their
default (fast) path, now count each pixel in the lookup-table entry it reads: its 8-bit level `(x * 255).long()` in
the input dtype. Before, they counted it in `floor(256 x)`, the `torch.histc` placement, which put every value in
`[k/256, k/255)` one bin above the entry it reads; that is about half of a continuous image (#6181). `equalize` of a
float32 or float64 image is now `PIL.ImageOps.equalize` of those levels, as long as no level of a float32 channel
holds more than 2**24 pixels. A channel whose lookup step is 0 is still returned unchanged.

On continuous float32 input, over 20 random images per size:

- `equalize` changes 48–49% of the pixels:
  - at 64 x 64, 42% move by 1 gray level and 1.8% by 2;
  - at 300 x 300, 49% move by 1 and 0.04% by 2;
  - in one 64 x 64 image the lookup step went from 16 to 15, and its pixels moved by up to 18 levels.
- `equalize3d` changes 50% of the voxels of 4 x 64 x 64 volumes.
- `equalize_clahe` with the default arguments changes about 77% of the pixels at 256 x 256 and 296 x 296. The pixels
  that change move by 0.73 gray levels on average, and by at most 3.6.

8-bit images in float32 and float64 are unchanged. On 8-bit images, half-precision `equalize_clahe` now stays within
0.2 gray levels of float32 in float16 and within 1.5 in bfloat16 (default arguments, 256 x 256 and 296 x 296).
