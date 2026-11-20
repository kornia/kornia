Preserve the documented first-source gradient of `distance_transform` under `torch.vmap`, and allow forward derivatives through batched distance transforms, including inside `torch.no_grad()`.
