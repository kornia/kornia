`find_homography_dlt`, `find_homography_dlt_iterated`, `find_homography_lines_dlt`, `find_homography_lines_dlt_iterated`,
`find_fundamental`, `run_7point`, `run_8point` and `RANSAC` run with autocast disabled, so a caller's `torch.autocast`
region no longer changes their result. Under bfloat16 autocast a float32 input used to be fitted with its matrix products
in bfloat16 and returned as bfloat16: the iterated DLT homography of 500 points with 1 px noise moved the corners of a
600 x 450 image by 3 to 5 px, a `RANSAC("homography", local_optimization="dlt")` estimate was about 3 px off instead of
0.5 px, and `find_fundamental` returned a bfloat16 matrix far from the float32 one. On CUDA, the 8-point
`find_fundamental` with more than eight points or with weights, and `RANSAC("fundamental", local_optimization="dlt")`,
raised `NotImplementedError` from `torch.linalg.eigh`. The default `local_optimization="lm"` solved and scored its minimal
samples in the autocast dtype, which could change its estimate.
