EfficientViT keeps its linear attention in float32 for float16 inputs under CPU autocast, preventing NaNs in pretrained feature maps.
