`find_homography_lines_dlt`, `find_homography_lines_dlt_iterated` and `find_homography_dlt(..., solver="svd")`
return NaN for a batch element whose DLT system is not finite, such as one with a NaN or infinite point, endpoint or
weight, and the usual homography for the rest of the batch, on CPU and CUDA, eager and compiled. On CPU, one such
element made `torch.linalg.svd` raise for the whole batch; the solvers then warned "SVD did not converge" and returned
uninitialized memory from `torch.empty`. Every homography of the batch was then garbage that changed from call to call
and had no gradient. Compiled and exported programs raised `LinAlgError` instead. On CUDA, a non-finite weight could
give a finite but meaningless homography. If the SVD fails on a finite system, the whole batch is now NaN, with the
same warning.
