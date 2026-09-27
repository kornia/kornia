`render_gaussian2d` no longer adds `1e-8` to each axis's normalising sum, so a Gaussian inside the image now sums to
1 up to roundoff instead of `1 - 1.1e-8`. When every sample on an axis underflows (the mean lies far off the image),
the heatmap stays all zeros, as before, with finite gradients.
