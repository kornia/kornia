`KimiVLBuilder.from_pretrained_hf()` and `SigLip2Builder.from_pretrained_hf()` cache their
checkpoint where every other kornia checkpoint lives. `cache_dir=None` used to mean the HuggingFace
cache (`~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<sha>/model.safetensors`) and now
means torch's hub directory (`<torch hub dir>/checkpoints/<owner>--<name>--model.safetensors`); an
explicit `cache_dir` used to hold the same `models--<owner>--<name>/…` tree and now holds the flat
`<owner>--<name>--model.safetensors` file. Either way the first `from_pretrained_hf()` call after
upgrading re-downloads the checkpoint (854 MB for KimiVL, ~1.5 GB for SigLIP2) even for users who
already had it. (#4293)
