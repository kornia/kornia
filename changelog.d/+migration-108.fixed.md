Make YUV and XYZ transformations compute integer inputs in `float32` instead of truncating their kernels, and
preserve directly constructed `float64` coefficients. This makes `rgb_to_yuv(uint8)` return `float32` on the
input's original 0--255 scale, fixes signed-integer YUV results, and removes the existing float32 coefficient loss
from float64 XYZ conversions. The two private linear-transformation helpers are now one implementation, whose CPU
path uses its dtype/device-aligned compute operands (#4053).
