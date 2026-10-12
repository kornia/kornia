`Z1Projection.unproject` now materialises a python `int`/`float` `depth` on the device and
in the dtype of `points`; it used to build a CPU float32 tensor, which raised `RuntimeError`
on every accelerator and widened float16/bfloat16 results to float32. (#4340)
