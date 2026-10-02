`mean_iou_bbox` now computes float16 and bfloat16 box areas in float32, returning finite overlaps for ordinary image-sized boxes while preserving the output dtype.
