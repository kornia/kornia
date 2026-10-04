`Quaternion` now registers plain tensor data as a persistent buffer, preserving autograd history
and explicitly supplied `nn.Parameter` inputs at construction. Previously, plain tensor rotations
were omitted from module state; `Quaternion`, `So3` and `Se3` now save and restore them
and follow enclosing module device/dtype conversions. This adds `_data`, `_q._data`
and `_rotation._q._data` checkpoint keys, respectively. Older checkpoints missing these
keys now fail with `strict=True`; `strict=False` reports the missing keys and retains
the target's initialized rotation, which must be supplied separately if needed.
