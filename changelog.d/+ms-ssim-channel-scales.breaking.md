`MS_SSIMLoss` now pairs every channel with every scale. Previously its masks were laid out scale-major while the grouped
convolution reads them channel-major, so for RGB input red was filtered at scales 0, 0, 0, 1, 1, green at 1, 2, 2, 2, 3
and blue at 3, 3, 4, 4, 4, and the luminance term saw only blue: brightening the red or green channel by 0.3 cost
almost nothing while brightening blue cost 0.28, and swapping RGB to BGR changed the loss. Single-channel input got the
MS-SSIM term cubed, and inputs whose channel count does not divide `3 * len(sigmas)` (2 or 4 channels with the default
sigmas) raised `RuntimeError`. The MS-SSIM term is now one minus the product, over the channels, of the per-channel
MS-SSIM (luminance at the coarsest scale, contrast-structure at every scale), any channel count is accepted, and loss
values change for every input. The masks are rebuilt from `sigmas` instead of being persisted: old checkpoints still
load strictly, but their `_g_masks` values are ignored.
