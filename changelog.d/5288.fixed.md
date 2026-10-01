`Resize` with an integer size, `LongestMaxSize` and `SmallestMaxSize` now return an empty batch with the
resized spatial shape for a `(0, C, H, W)` input instead of raising `KeyError: 'output_size'`.
