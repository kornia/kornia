`RandomJPEG` accepts an empty batch and returns it unchanged. It raised
`ValueError: Invalid input tensor, it is empty.` before the codec ran, because the differentiable JPEG codec rejects
an empty input.
