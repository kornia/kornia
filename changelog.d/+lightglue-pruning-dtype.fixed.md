`LightGlue` and `LightGlueMatcher` with point pruning on (the default `width_confidence=0.99`, and the `superpoint`,
`disk`, `aliked`, `raco-aliked`, `dedodeb`, `dedodeg`, `sift` and `xfeat` feature types) raised
`RuntimeError: Index put requires the source and destination dtypes match` for float64, float16 and bfloat16 models,
and for float32 models with `mp=True` on CPU, where autocast computes the scores in bfloat16. On CPU and MPS this
happened at any number of keypoints, on CUDA above the pruning threshold. Pruning now keeps the dtype the scores are
computed in. The two early returns, for no descriptors on one side and for every point pruned, returned float32
tensors for such inputs and now return them in the input dtype. float32 results are unchanged. (#5837)
