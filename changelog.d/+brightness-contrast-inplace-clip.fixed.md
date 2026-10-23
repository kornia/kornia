`adjust_brightness` and `adjust_contrast`, and with them `RandomBrightness`, `RandomContrast` and `ColorJiggle`,
clamp their own freshly computed result in place when no gradient is recorded. That saves one image-sized temporary
per call. At batch 32 of 3×256×256 float32 on CPU, the extra 25 MB buffer made glibc trim and re-fault the heap on
every call (18,400 page faults per call against 12,256), and `RandomBrightness` took 8.6–9.7 ms against 6.2 ms in
0.8.3. It now takes 7.5 ms, the remaining cost being the `[0, 1]` input clamp added in #5980. Values, gradients and
the caller's tensor are unchanged.
