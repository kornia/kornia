`connected_components` now returns `int64` labels with exact ids. It used to return the input dtype, which merged distinct components in `float16`, `bfloat16`, `bool` and `uint8` inputs (ids lost precision or wrapped) and, in any dtype, above 2**24 pixels. On MPS, which has no `float64`, an image of more than 2**24 pixels (a single image, not the batch) now raises `ValueError` instead of returning merged labels.

`KMeans(tolerance=None, max_iterations=0)`, a negative `max_iterations` and a negative `tolerance` now raise `BaseError`. The first never stopped fitting. A negative `max_iterations` used to stop the fit after one iteration, and a negative `tolerance` used to be accepted and never triggered the early stop.

`KMeans.predict` with a 1-D input now raises `ShapeError` instead of `IndexError`.

`distance_transform` and `DistanceTransform` now reject an `h` so small that the kernel's corner tap `exp(-sqrt(ndim) * (kernel_size // 2) / h)` underflows, which the old axis-tap check let through. Such calls used to return wrong distances (for example float32 17.78 instead of 21.21 at `kernel_size=31`, `h=0.18`).

`distance_transform` and `DistanceTransform` now reject an `h` above a bound that depends on `kernel_size`: in 2-D `h < 0.5626` at `kernel_size=3` and about `0.50` at larger kernels, in 3-D `h < 0.4071` at `kernel_size=3` and about `0.382` at larger kernels. Such an `h` used to return 0, as if on the foreground, at background pixels whose window was full of reached pixels.
