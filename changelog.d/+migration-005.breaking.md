`SegmentationModelsBuilder.build()` no longer imports `segmentation_models_pytorch` (smp). It used
to take `model_name`, `encoder_name`, `encoder_weights`, `in_channels`, `classes`, `activation` and
`**kwargs`, import smp lazily, instantiate the smp architecture and look up the encoder's
preprocessing parameters itself. It now takes a constructed `nn.Module` and the encoder's
preprocessing-parameter dictionary (`build(model, preproc_params=None, name="segmentation_model")`),
where `preproc_params` is what `smp.encoders.get_preprocessing_params(encoder_name)` returns; build
the smp network and fetch its parameters yourself. Kornia no longer imports smp anywhere, and
`kornia.core.external.segmentation_models_pytorch` is gone. `SemanticSegmentation` gained the
`__init__(model, pre_processor, post_processor, name=None)` its siblings have, so the container the
builder returns can be instantiated (it was abstract before). The `input_range: [0, 255]`
preprocessing step is now `kornia.enhance.Rescale(255.0)`, a multiply by 255, instead of a
`Normalize` by a stored `1/255`: bfloat16 rounds that reciprocal to a ~254.0 multiplier (0.5 mapped
to 127.0 instead of 127.5), and float32 outputs of that step move by at most one ulp. (#4301)
