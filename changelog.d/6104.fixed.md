`focal_loss` and `FocalLoss` raise a `RuntimeError` for a label outside the classes under `torch.compile` on the CPU, as
they do in eager mode. With `reduction='mean'` or `'sum'` and no `weight`, the loss gathers the target class, and
Inductor's CPU gather checks its indices inside an OpenMP parallel region: a label past the classes terminated the whole
Python process (exit code 134, "index out of bounds") on torch 2.5.1, 2.9.1 and 2.14, a negative label did the same on
2.14, and 2.5.1 and 2.9.1 returned a loss for a negative label when no gradient was needed. The labels are now checked
with `torch._assert_async` before the gather, and on torch 2.14 a compiled gather reads them clamped into range, so it
stays in bounds until the check raises. Inductor before torch 2.9 compiles the assert to nothing, so there a compiled
call still fails as before. The check uses `one_hot`'s messages: an eager CPU call now raises "Class values must be
non-negative." or "Class values must be smaller than num_classes." where the gather raised "index ... is out of bounds
for dimension 1 with size C", as the `'none'` and weighted losses already did. CUDA is unchanged: an out-of-range label
still fails the gather's device-side assert there.
