Make `guided_blur` / `GuidedBlur` work in `float16` and `bfloat16` when the guidance has more than one channel.
Multi-channel guidance solves a per-pixel `C x C` system, and `torch.linalg.solve` has no half-precision LU
kernel, so every such call failed: on CPU with `NotImplementedError: "lu_cpu" not implemented for 'Half'`, and on
MPS with a hard `Only MPSDataTypeFloat32 is supported` assertion that aborted the process rather than raising.
The system is now solved in `float32` and cast back. A tensor `eps` also takes part in dtype promotion, so
the common `eps=torch.tensor(0.1)` (float32) widened the matrix past the right-hand side and made `solve`
reject the pair; both operands are now solved in one dtype. Single-channel guidance never reached the solver
and is unaffected. `float32` and `float64` results are bit-identical to before, and the op still compiles with
`fullgraph=True` in one graph. This takes `tests/filters/test_guided.py` from 98 failed / 67 passed to
169 passed under `--dtype=float16,bfloat16`.
