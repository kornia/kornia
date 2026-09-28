Fix `symmetric_transfer_error` returning `NaN` instead of `max_num` for singular homographies.
When `safe_inverse_with_mask` flags an input as non-invertible, the unshielded inverse matrix containing
non-finite values propagated `NaN` through `oneway_transfer_error` and the arithmetic mask
`(there + back) * mask + max_num * (~mask)`, poisoning both the forward error and autograd gradients.
Singular rows are now replaced by the identity *before* the differentiable inverse is taken, so the
gradients with respect to both the correspondences and `H` itself stay finite (zero on the singular
rows), and the final masking uses `torch.where` to cleanly assign `max_num`. Values for invertible
homographies are unchanged; for `squared=False` a singular row now scores `max_num` rather than the
previous `(max_num + eps).sqrt()`, matching the documented contract. (#4203)
