`AugmentationSequential` with `mask`, box or `keypoints` data keys, and the `ManyToManyAugmentationDispather` and
`ManyToOneAugmentationDispather` built from it, export again with torch 2.9's dynamo ONNX exporter when replayed with
constant `params` and a `RandomAffine` child. Since #5416 the annotation handlers recompute the child's matrix from the
caller's params under export, and `RandomAffine` copies `center` from them as-is, which that exporter turns into a
renamed graph constant and rejects. The handlers now compute from a copy of the params, as the image call does.
Eager outputs are unchanged. Fixes #6166.
