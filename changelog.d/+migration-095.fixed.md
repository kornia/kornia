`Boxes3D.to_tensor` and `Boxes3D.get_boxes_shape` are differentiable again (#1396). Both reduce a box's 8
vertices to its min/max corner with `amin`/`amax`, which is a genuine kink -- not differentiable in the
classical sense -- wherever multiple vertices exactly tie for an axis extremum. Every face of an
axis-aligned box has such a 4-way tie by construction, where PyTorch's `amin`/`amax` backward splits the
gradient evenly across the tied vertices (`1/4` each): a valid subgradient, usable for optimization, but
one that `torch.autograd.gradcheck`'s central-difference estimate does not match at that exact point --
the same non-uniqueness any reduction has at a kink (`torch.max`, `ReLU`). That mismatch was previously
taken for a genuine gradient bug and `to_tensor` raised `RuntimeError` whenever its input required grad,
unconditionally, including for the common case of a caller who never differentiates through a tie at all
(an oriented, non-axis-aligned box, or a loss that never reaches the tied component). The guard is
removed; the docstring documents the tie behavior instead of calling it a bug. `Boxes` (2D) uses the same
reduction and was never guarded, because an axis-aligned rectangle's ties are always 2-way, where the
`1/2` split happens to numerically coincide with the central-difference estimate -- the same kink, just
invisible to `gradcheck` there.
