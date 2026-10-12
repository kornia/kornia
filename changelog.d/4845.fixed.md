`confusion_matrix` and `mean_iou` return an empty `(0, K, K)` / `(0, K)` result for a zero-length batch instead of raising `RuntimeError` on `view(0, -1)`.
