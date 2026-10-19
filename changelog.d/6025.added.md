`kornia.models.qwen25.Qwen25VisionTransformer` and `Qwen25VisionConfig`, the Qwen2.5-VL vision tower: a
temporal patch embedding, a 2D rotary position embedding, RMS-normalized blocks with gated MLPs, windowed
attention with full attention in selected blocks, and the 2 x 2 patch merger. `from_pretrained` strictly
loads the `visual.*` tensors of a Qwen2.5-VL checkpoint from a local file or directory. On the released
3B weights it matches Hugging Face `transformers` v4.51.0 on the same preprocessed patches.
`kornia.core.load_safetensors` gains an `include` predicate that copies only the selected tensors out of
the file, so a vision tower can be read from a shard it shares with a language model. Closes #6003.
