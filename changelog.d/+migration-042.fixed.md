`VideoBoxes.get_boxes_shape()` and `VideoBoxes.to_mask()` no longer raise `TypeError`. Both are
inherited from `Boxes` and pass `as_padded_sequence=True` to `to_tensor`, but the `VideoBoxes`
override declared only `mode`, so every call on a `VideoBoxes` failed -- including the containers
`AugmentationSequential` builds for video box inputs. The override now accepts and forwards the
keyword; it only changes a list-backed container, and a `VideoBoxes` built from a
`(B, T, N, 4, 2)` tensor is not one, so no working call changes its result. Indexing a
`VideoBoxes` still drops `temporal_channel_size`, which is the remaining half of #4249.
(#4176, #4365)
