`BoxBlur` now stores writable kernel buffers so `load_state_dict` can restore both separable and non-separable modules without an overlapping-memory error.
