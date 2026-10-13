`DeDoDe.detect` and `DeDoDe.describe` now work under `torch.compile`. They were decorated with
`@torch.inference_mode()`, and inductor failed on the inference tensors they produce: with
`RuntimeError: Cannot set version_counter for inference tensor` on torch 2.14, and with the tensor-storage error on
torch 2.9.1 that the export-support page reported as an inductor bug. They now run under `@torch.no_grad()`, like
`DeDoDe.forward` compiled before. Gradients stay off and the eager outputs have the same values; they are ordinary
tensors now instead of inference tensors, so they can be used in autograd or modified in place afterwards. The compiled
`detect` can return keypoints with equal scores in a different order from eager.
