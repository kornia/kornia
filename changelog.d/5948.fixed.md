`render_gaussian2d` now evaluates half-precision Gaussian parameters in float32, avoiding NaN heatmaps and gradients for narrow float16 Gaussians while retaining the output dtype.
