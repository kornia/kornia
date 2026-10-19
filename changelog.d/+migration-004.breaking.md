`kornia.contrib.BoxMotTracker` is removed, together with the `kornia.contrib.boxmot_tracker` module.
It wrapped the [boxmot](https://github.com/mikel-brostrom/boxmot) tracker zoo around a kornia detector,
but called the boxmot 10.x API (`boxmot.DeepOCSORT(model_weights=..., device=..., fp16=...)`), and no
boxmot release that installs alongside the torch versions kornia supports (`torch>=2.5.1`) provides
it, so the class could not be instantiated on any supported stack (see
[#4320](https://github.com/kornia/kornia/issues/4320)). Track with boxmot directly, feeding it the
output of a kornia detector such as `RTDETRDetectorBuilder`. `kornia.core.external.boxmot` is gone
with it. (#4301)
