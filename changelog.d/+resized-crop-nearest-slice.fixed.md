`RandomResizedCrop(resample="nearest")` no longer raises in the default slice mode. Slice mode now ignores
`align_corners` for nearest resampling, for images and masks. (#4802)
