`undistort_points_kannala_brandt` no longer collapses representable `float16` points next to the principal point
to the origin, and the exact principal-point path now has finite autograd gradients. The zero-radius decision is
made from the unsquared normalized coordinates, and the `float16` radius uses `float32` intermediates so its
squared value does not underflow; nonzero radial rescaling remains epsilon-free. (#4308, #4370)
