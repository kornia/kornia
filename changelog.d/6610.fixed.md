`HausdorffERLoss` and `HausdorffERLoss3D` now erode a float64 prediction with a kernel that `get_kernel` builds in
float64 instead of the `kernel` buffer, and `MS_SSIMLoss` builds its Gaussian masks in float64 whenever it filters in
float64 (float64 images or a float64 module). They used to cast weights built once in float32 (0.2, 1/7 and the
Gaussian masks), so float64 results carried float32 rounding and differed from the same loss constructed under a
float64 default dtype. A float64 `MS_SSIMLoss` given float32 or half-precision images therefore now returns the float64
result rounded to the images' dtype, which can differ slightly from before; other losses computed in float32 or half
precision are unchanged. A subclass that overrides `get_kernel` must accept the new `dtype` argument to erode float64
predictions.
