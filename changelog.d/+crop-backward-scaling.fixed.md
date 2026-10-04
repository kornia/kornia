`crop_by_indices` avoids full-batch gradient buffers for each image crop, reducing backward memory traffic for slice-mode crop augmentations.
