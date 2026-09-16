`kornia.filters.kernels.gaussian` no longer returns NaN when `sigma` is small relative to the
distance from the mean to its nearest sample, or exactly zero. The kernel is now normalised in log
space and degrades to an impulse at the nearest sample(s). This fixes `RandomGaussianIllumination`,
which accepts `sigma=0` and turned every pixel into NaN for zero or small sampled sigmas.
