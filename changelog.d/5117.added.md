`RANSAC(compile=True)` runs the whole `local_optimization="lm"` estimation (homography, fundamental and essential
matrices) as one `torch.compile` graph on torch 2.14 or later. The graph is traced once per configuration and does not
depend on the number of correspondences, the threshold, the confidence or the sample budget; it is saved next to
inductor's cache, so a later process loads it instead of compiling. Seeded compiled calls are reproducible but use
their own random stream. Eager results are unchanged.
