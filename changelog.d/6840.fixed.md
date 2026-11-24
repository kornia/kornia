Keep box IoUs in float32 when matching float16 or bfloat16 detections in `mean_average_precision`, so overlaps just above the threshold count as matches.
