`SOLD2` and `SOLD2_detector` converted to float16 with `.half()` now detect line segments. Two BatchNorm layers of
the pretrained hourglass backbone have running variances up to 2.6e6, beyond float16's range: converted to float16
they became inf, the heatmaps collapsed and no segment was found. When the model is converted to float16, those
layers now keep their running statistics in float32 and normalize in float32. A model that is not converted to
float16 gives the same results as before, also under `torch.autocast`.
