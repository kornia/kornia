`ImageRegistrator.register` now optimizes a float16 or bfloat16 model or image in float32 and hands the model back in
its own dtype. It used to optimize in the half dtype, where Adam's second-moment estimate and `eps` round to 0 for
gradients below about 0.008: a float16 similarity registration ended in NaN parameters on CPU and raised in
`linalg.inv` on CUDA, and half registrations stopped short of the float32 result, such as -5.78 to -5.85 degrees
instead of -6.00 for a 6 degree rotation.
