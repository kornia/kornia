`RandAugment`'s `m` guard is exclusive at both ends, but its docstring and its error message
both named the closed interval `[0, 30]`, so a user who asked for the maximum strength the
message advertised got an exception saying `30` was in range. Both now read `(0, 30)`; the
accepted values are unchanged. `n` was validated nowhere: `n=0` constructed a `RandAugment`
that applied nothing, and `n` above the policy length was silently clamped, because the
sampler draws without replacement. It is now checked against the policy. `AutoAugment`'s
magnitude bin indexes two adjacent points of an 11-point scale, so `9` is the last usable
bin; a larger one raised a raw `IndexError` naming an internal tensor, and a negative one
wrapped silently onto a reversed range. Out-of-range bins now name the operation and the
valid range. Operations that ignore the magnitude entirely keep accepting any bin. (#4447)
