`otsu_threshold` and `OtsuThreshold` return different thresholds, and with `slow_and_differentiable=True` a
differentiable one:

- The threshold is now the upper edge of the last histogram bin below Otsu's split,
  `min + (t + 1) * (max - min) / nbins`, the bin edge `torch.histc` itself uses. It used to be read from
  `linspace(min, max, nbins)`, whose spacing is `(max - min) / (nbins - 1)`, which put it up to one bin above the
  split. Thresholds move down by up to one bin: on a `uint8` image with values 0 to 255 the threshold can drop from 126
  to 125, the value scikit-image and OpenCV return, and the pixel valued 126 is now kept. With `nbins=2` the
  threshold used to be the data maximum, so no pixel was ever kept.
- Each `(b, c)` plane is now histogrammed and thresholded on its own minimum and maximum. All planes of a call used to
  share the range of the whole input, so a plane's threshold, and the pixels it kept, depended on the other images and
  channels in the batch.
- A constant plane now gets its own value as threshold, so none of its pixels is kept. It used to get the threshold 0,
  which kept every pixel of a positive constant plane and none of a negative one.
- With `slow_and_differentiable=True` the threshold now has a gradient with respect to the input. Its value is the
  Otsu split of the slow path's histogram, as above, and its gradient that of a soft-argmax over the between-class
  variance curve (a straight-through estimator). That histogram is now the mass a Gaussian kernel density estimate of
  bandwidth 0.1 bin puts in each bin, so every pixel contributes. It used to sample the estimate, with a fixed
  bandwidth of `1e-3` in input units, at `nbins` points, which missed most pixels at small `nbins`, and the threshold
  had no gradient. The slow path's thresholds change accordingly and are usually within one bin of the default
  path's. The thresholded image is still `x * (x > threshold)` in both modes, with that mask as its gradient.
- float16 and bfloat16 inputs are histogrammed and thresholded in float32. The counts used to come back in the input
  dtype, so a float16 plane of 65520 pixels or more got the threshold 0, and the split arithmetic ran partly in half
  precision.
