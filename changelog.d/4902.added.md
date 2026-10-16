`RANSAC(prosac_sampling=True)` now draws progressive samples from correspondences sorted best-first, following Chum
and Matas (CVPR 2005); the flag used to be accepted and ignored. The schedule advances per sampled set inside each
batch, and the sampler runs the full `batch_size * max_iter` budget. `RANSAC` also accepts `lo_sample_size`, which
fits `max_lo_iters` random inlier subsets of that size in one solver batch before a final full-inlier refit.
