`RandomCrop3D` and `CropGenerator3D` now reject a fixed crop size larger than the input volume for empty batches, consistently with non-empty batches. Crops that fit after padding remain valid.
