Fix `DenseSIFTDescriptor` registering a process-global cached tensor as its `_poolingconv_weight` buffer, so that
one instance's `load_state_dict` silently corrupted every other instance and every instance constructed later
(#4068). `_get_reshape_kernel` memoises `torch.eye(numel)` per `numel` and used to return a *view* of the cached
tensor; `.float()` on an already-float tensor returns the same object, so no copy happened between the cache and
the buffer, and `load_state_dict` copies into buffers in place. Loading a checkpoint into one `DenseSIFTDescriptor`
therefore overwrote the shared identity matrix, with no error raised and every descriptor built from it wrong from
then on. The cache is removed rather than made safe: cloning on the way out is what a safe cache requires, and a
clone costs more than rebuilding the identity at every size measured (2.6 us against 2.0 us at the default
`numel = 8 * 4 * 4`; 3.9 ms against 2.8 ms at the former 4096 bound, which also retained 67 MB for the lifetime of
the process). Descriptor output is byte-identical to before.
