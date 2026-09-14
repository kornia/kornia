`add_weighted` now keeps Python scalar weights in operator math precision instead of rounding them to the input
dtype before arithmetic. This improves `float16` and `bfloat16` accuracy and makes fractional scalar weights on
integer inputs produce the correctly promoted floating-point result instead of being truncated. (#4154)
