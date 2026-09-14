KimiVL and SigLIP2 builders load their safetensors checkpoints with kornia's own downloader
(`kornia.core.download_hf_file`/`download_file_from_url`, which share the retrying, rate-limit-aware
cache every other checkpoint uses) and a pure-torch reader (`kornia.core.load_safetensors`);
`huggingface_hub` and `safetensors` are no longer needed. Both were imported without ever being
declared as dependencies, so `KimiVLBuilder.from_pretrained_hf()` and
`SigLip2Builder.from_pretrained_hf()` used to raise `ImportError` on a plain `pip install kornia`.
The checkpoints move to torch's hub cache as a result; see *Breaking changes*. (#4293)
