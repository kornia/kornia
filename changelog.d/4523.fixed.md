Extend the CUDA `torch.compile` constant-transfer workaround to shear, thin-plate spline, erasing, crop, resize, 3D affine/perspective, and illumination/mix parameter generators. Shared constant construction preserves eager parameter placement and avoids extra eager scalar allocations. Crop operations retain their existing graph breaks.

Fix `RandomShear` sampler migration so scalar and tuple ranges honor the requested device and dtype. Fix mixed-device sign generation in both linear-illumination generators after moving their samplers.
