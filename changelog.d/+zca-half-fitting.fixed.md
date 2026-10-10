ZCA whitening fits float16 and bfloat16 data in float32 to avoid overflowing the covariance accumulation, while preserving the input dtype of the fitted mean and transforms.
