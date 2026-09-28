`kornia.io.get_sample_images` and `kornia.onnx.ONNXLoader.list_operators` / `list_models` no longer need
`requests`, which `pip install kornia` never installed; they fetch with the standard library `urllib` instead.
The failure exceptions change with it, all still `OSError` subclasses: a sample-image URL that returns 404 now
raises `urllib.error.HTTPError` instead of `PIL.UnidentifiedImageError`, and an unreachable host raises
`urllib.error.URLError` instead of `requests.exceptions.ConnectionError`. A 404 from the Hugging Face listing
still raises the same `ValueError` (#4302)
