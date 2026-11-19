`VideoSequential` now broadcasts shared Gaussian noise across all frames when `same_on_frame=False` and `same_on_batch=True`, avoiding a batch-size error for multiple videos.
