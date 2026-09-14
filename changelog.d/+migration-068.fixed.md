The generated example figures in the API reference are rendered from RGB input.
`docs/generate_examples.py` decoded the sample images with `cv2.imdecode` and wrote them with
`cv2.imwrite`, both BGR, so every tensor the examples fed to kornia held reversed channels. The two
swaps cancel in the written file for channel-agnostic operations, but 35 colour-dependent figures
change, and the ones that depend on which channel is which (`rgb_to_hsv`, `ColorJitter`,
`apply_colormap`, the bbox and keypoint colours of `AugmentationSequential`, ...) showed a blue cast
or swapped overlay colours. The generator now decodes and writes with PIL, draws the KeyNetAffNet
LAF figure with kornia's own `get_laf_pts_to_draw`, and `opencv-python` and `kornia_moons` leave
the `[docs]` extra. (#4306)
