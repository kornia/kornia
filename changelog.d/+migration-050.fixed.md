`Normalize`, `Denormalize` and `Rescale` register their constants (`mean`, `std`, `factor`) as
non-persistent buffers instead of plain attributes, so `.to(device)` moves them with the module.
Previously they stayed on the CPU: eager tolerates the mix, but `torch.export` traces with fake
tensors and refused it, so exporting a preprocessing pipeline from an accelerator failed while
the same pipeline exported fine from the CPU. `Denormalize` now coerces a scalar `mean`/`std` to
a 1-D tensor, as `Normalize` already did, which changes its `__repr__` to match `Normalize`'s and
lets a scalar `Denormalize` reach the ONNX export branch instead of raising `IndexError` there.
The buffers are non-persistent, so `state_dict()` is unchanged and existing checkpoints still
load. (#4323, #4330)
