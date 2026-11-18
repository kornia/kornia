`crop_by_indices` retains the image gradient link for empty batches, fixing backward through empty slice-mode `CenterCrop2D`, `RandomCrop`, and `RandomResizedCrop` calls.
