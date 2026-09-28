`PatchSequential` now preserves the batch size in both padding modes. `same` preserves the input
spatial size and `valid` keeps only the complete, centred grid region. Previously, a `(2,3,8,8)`
input on a `(3,3)` grid produced `(2,3,7,7)` with `same` (now `(2,3,8,8)`) and `(2,3,8,8)` with
`valid` (now `(2,3,6,6)`). With `valid`, a `(2,3,6,8)` input on a `(4,4)` grid changed the batch
size to produce `(3,3,6,6)`; it now produces `(2,3,4,8)`. Geometric children may move temporarily
padded zeroes into the retained image. `restore_from_patches` now defaults to `self.grid_size`
instead of `(4,4)` and raises `ValueError` for a mismatched patch count instead of inferring a
different batch size or failing later in reshape. `forward_parameters` accepts both image
`(B,C,H,W)` and patch `(B,N,C,h,w)` shapes; parameter generation now covers `B*N` patch rows rather
than `B*C`, so fixed-seed outputs and RNG consumption can change. Refs #4421. (#4460)
