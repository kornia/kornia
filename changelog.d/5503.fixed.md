Descriptor matching now computes half-precision distances in float32 to avoid overflow and lost nearby matches, while returning distances in the input dtype.
