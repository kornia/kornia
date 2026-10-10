`HardNet8()` without pretrained weights now gives distinct, trainable descriptors. Its placeholder PCA projection
was an all-ones matrix, which mapped every patch to the same descriptor up to sign; it now keeps the first 128 of the
512 features until the pretrained weights replace it.
