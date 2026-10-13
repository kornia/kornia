`kornia.filters.guided_blur` and `GuidedBlur` accept an empty batch with a multichannel guidance. They raised
`RuntimeError: cannot reshape tensor of 0 elements` because the input channel count was inferred with `-1`, which
an empty tensor cannot provide; this happened both when the guidance is the input and when it is a separate tensor.
