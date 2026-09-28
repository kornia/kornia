A fresh transfer that `validate` rejects is now removed even when no other source could have used
the emptied path -- which is every `download_hf_file` caller, since it passes one URL. Those bytes
arrived during the call and were refused, so unlike the ambiguous load failures
`load_state_dict_from_url` weighs, keeping them would end the call having added a poisoned entry to
a cache that had none. An offline re-fetch also reports the validator's rejection rather than the
network error stacked on top of it, so the message names the file rather than an entry that is
intact and, by then, restored. (#4367)
