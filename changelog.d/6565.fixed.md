`determinant_to_polynomial` now evaluates float16 and bfloat16 inputs in float32 and returns the input dtype. It used to
sum the coefficients in the half dtype, so on CUDA two calls on the same input could return different coefficients.
