Models that reshaped a batch with a `-1` next to the batch dimension now accept empty batches instead of raising
"cannot reshape tensor of 0 elements" (#6266): `TinyViT`, `VisionTransformer`, `YuNet` and `FaceDetector`, the SAM
image encoder and mask decoder (an image with no prompts), the EfficientViT backbones, and DINOv2's
`get_intermediate_layers(reshape=True)`. `window_partition` and `window_unpartition` accept an empty batch too.
