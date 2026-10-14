`connected_components` accepts batches that cannot be viewed as one block (for example permuted leading dimensions) and empty inputs, which used to crash.

`KMeans.fit` with `tolerance=0.0` now stops at the fixed point instead of never returning, and a second `fit` samples fresh initial centres from its own data (it reused the first fit's centres, and raised for data of another dimension) unless explicit `cluster_centers` were given. Distances in `fit` and `predict` are computed in at least `float32`, so `float16` squared distances no longer overflow to `inf` and pick cluster 0. `predict` checks that its input is 2-D.

`Lambda` names the right argument (`func`) in its non-callable error message.
