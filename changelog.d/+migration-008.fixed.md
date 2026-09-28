`bbox_to_mask3d` now preserves the input dtype in the returned mask, matching `bbox_to_mask`
and the `Boxes3D.to_mask` contract: the earlier implementation downcast every result to
`float32`, so a `float64` or half-precision box produced a mask that silently lost precision.
The mask keeps the input box's dtype end to end. (#4376)
