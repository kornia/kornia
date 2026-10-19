Non-maxima suppression with a window larger than `(7, 7)` no longer builds a `(k*k, 1, k, k)` one-hot
convolution to gather each neighbour into its own channel. On CPU in half precision that convolution
has no vectorized kernel and falls back to `_slow_conv2d_forward`, where it accounted for 99% of a
`MultiResolutionDetector.detect` call. The window minus its centre is now covered by rectangular
`max_pool2d` regions whose full-width slabs share one column pass, costing `O(ky + kx)` taps per
position rather than `ky * kx - 1`, with the centre excluded by construction so the suppression stays
strict. `detect` on a 240x240 image drops from 5757 ms to 27 ms in float16 and from 73.8 ms to 13.8 ms
in float32; a 353x353 `k = 15` suppression drops from 3150 ms to 10.5 ms in float16 and from 42.6 ms to
8.7 ms in float32, and from 12.1 ms to 0.15 ms for 1024x1024 `k = 21` on CUDA. (#4242)
