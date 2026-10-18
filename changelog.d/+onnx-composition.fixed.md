`kornia.onnx` composition and loading:

- `ONNXSequential`, and `HFONNXComunnityModel` / `DepthEstimation` built on it, chain models whose metadata differ for
  a key: a graph exported by another kornia release (its `version`), exports given different `additional_metadata`
  values, or an onnx-community model whose `source` differs from the kornia pre-processor's. They used to raise
  `ValueError: Can't merge models with different values for the same model metadata property`. The combined model
  holds one entry per key, at its first position, with the value of the last model that sets it; the models passed in
  are not modified.
- `ONNXSequential` applies `target_ir_version` and `target_opset_version` without `auto_ir_version_conversion=True`,
  as `ONNXModule` does; it used to ignore them silently. Without the flag, only the versions given are changed, and a
  model that already has them is not converted again (`ONNXModule` as well).
- `ONNXSequential` warns (`UserWarning`) when the tensor element types of an output and the input it feeds differ at
  a joint: the merged graph does not check a later model's declared input types, so such a chain runs on the earlier
  model's type, or fails later with ONNX Runtime's `Type Error`, as before. Declared dimensions are not compared.
- A malformed `io_maps` raises a `ValueCheckError` (a `ValueError`) that says `io_maps` holds one list of
  `(output, input)` name pairs per joint, where it used to raise `IndexError` (an entry missing) or
  `ValueError: Output K00-m is not present in g1` (a flat list of pairs, the form the ONNX guide showed). A pair with
  more than two names, whose extra names were ignored, now raises it too. An entry may still be any iterable of pairs
  (a `zip`, `dict.items()`, a generator, a numpy array) and `io_maps` anything indexed by the joint, and entries past
  the last joint are still not read. The `io_maps` annotations of `ONNXSequential`, `ONNXSequential.combine` and
  `HFONNXComunnityModel` are now `list[list[tuple[str, str]]]`, and the guide's example is a list of lists.
- `ONNXModule`, `ONNXSequential`, `kornia.onnx.load`, `ONNXLoader.load_model` and `ONNXLoader.load_config` take
  file paths as `os.PathLike` objects such as `pathlib.Path` (always a local file) and as `bytes`, and expand a
  leading `~` in a local path. A path object used to raise `ValueError: Invalid argument type` (`AttributeError` from
  the two loaders). An argument of another type now raises `TypeError` instead of `ValueError`.
- Calling `ONNXModule` or `ONNXSequential` takes CPU torch tensors of a dtype numpy has (not bfloat16) as well as numpy
  arrays, and still returns numpy arrays; a tensor used to raise ONNX Runtime's `RuntimeError: Input must be a list of
  dictionaries or a single numpy array`. A tensor on another device raises a `ValueError` that says to move it to the
  CPU, and a bfloat16 or sparse tensor a `TypeError` that names the input and how to convert it.
- `ONNXLoader.load_config` reads an existing local JSON file, which it used to report as not found ("You may set
  `download=True`"). An argument that is neither an HTTP(S) URL nor an existing file (a directory, for one) raises a
  `RuntimeError` that says so.
- The `ONNXModule` docstring documents its parameter `op`, one model, and `io_name_conversion` documents that it
  applies any mapping in place, returns the same model, and leaves initializers, `value_info` and subgraphs alone.
