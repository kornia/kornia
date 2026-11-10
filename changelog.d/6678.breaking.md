`HausdorffERLoss` and `HausdorffERLoss3D` no longer ignore misclassified pixels whose true-class probability is between
0.29 and 0.5. The error of a pixel is its squared error above 1/4, where thresholding the probability at 0.5 starts to
misclassify it, rescaled so that a fully wrong pixel gives 1. It is weighted by `sum(i ** alpha for i in range(1, min(k,
d - 1) + 1))` for its city-block distance `d` to the nearest correctly classified pixel, counted by eroding the
misclassified set with the minimum over the pixel and its four (six in 3D) neighbours; the image border is not an edge.
The losses used to convolve, soft-threshold at 0.5 and min-max normalise every erosion per image, which gave zero loss
and zero gradient whenever every pixel's true-class probability was above 0.29, wrong predictions and a uniform two- or
three-class softmax included (#6676). Loss values change: a correct pixel now adds nothing however uncertain, and errors
at most two pixels wide add nothing. Only the nonzero entries of `get_kernel` matter now, as the structuring element,
and the `conv` and `max_pool` class attributes are removed.
