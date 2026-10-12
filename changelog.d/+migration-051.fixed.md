`download_file_from_url` and `download_hf_file` take an optional `validate=` callable, and
`kornia.core.check_safetensors` is the one the KimiVL and SigLIP2 builders pass. A transfer cut
short after a 2xx status leaves a truncated file in the cache, which was then returned as a cache
hit on every later call -- `from_pretrained_hf()` failed until the user deleted it by hand.
`validate` gives a download-only call the quarantine `load_state_dict_from_url` gets from its load
step: a rejected entry is moved aside, the next source is tried, and the discarded source is
re-fetched once. Without `validate` the behaviour is unchanged. (#4309, #4332)
