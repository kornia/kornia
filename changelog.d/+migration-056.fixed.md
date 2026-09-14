`ModelBase.save` and `_save_outputs` write the visualizations they are given instead of raising.
They handed `visualize`'s float output to `write_image` under a hard-coded `.png` name, but PNG is
`uint8`/`uint16` only, so every container's `save` (`SemanticSegmentation`, `ObjectDetector`,
`EdgeDetector`, `DepthEstimation`) raised on its first write and left an empty directory behind.
Float images are now converted to `uint8`, clamped to `[0, 1]` first so an out-of-range
visualization does not wrap. A batched `(B, 3, H, W)` output -- the shape the containers document
-- is written as one file per item rather than passed whole to `write_image`, which takes
`(3, H, W)` and rejected the rank with a message naming neither. (#4322)
