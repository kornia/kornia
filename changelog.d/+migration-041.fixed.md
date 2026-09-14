`warp_affine`, `warp_perspective` and `remap` crashed on MPS for an empty destination -- a `dsize` with a
zero dimension, or zero-sized `remap` maps -- with an internal
`[srcBuf length] > 0 INTERNAL ASSERT FAILED ... Placeholder tensor is empty!` from PyTorch. The MPS backend
rejects *any* zero-element `grid_sample` operand before torch 2.14, including a zero-element grid sampled
against a non-empty source, which is what the empty-destination path built. That path now samples a connected
1x1 stand-in and expands the result to the requested empty shape, so `grid_sample` never receives a
zero-element operand on any backend and an empty warp costs the same whatever the non-zero side of `dsize` is.
The empty path also resolves its output batch the way each operation's non-empty path does, so a singleton
transform or map batch broadcasts identically and a mismatched batch is rejected rather than silently
returning the wrong cardinality. Outputs, autograd links and the documented empty-source policy are unchanged
on CPU and CUDA. (#4032, #4354)
