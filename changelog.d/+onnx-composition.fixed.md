`kornia.onnx` composition and loading:

- `ONNXSequential`, and `HFONNXComunnityModel` / `DepthEstimation` built on it, chain models whose metadata differ for
  a key: a graph exported by another kornia release (its `version`), exports given different `additional_metadata`
  values, or an onnx-community model whose `source` differs from the kornia pre-processor's. They used to raise
  `ValueError: Can't merge models with different values for the same model metadata property`. The combined model
  holds one entry per key, at its first position, with the value of the last model that sets it; the models passed in
  are not modified.
- `ONNXSequential` applies `target_ir_version` and `target_opset_version` without `auto_ir_version_conversion=True`,
  as `ONNXModule` does; it used to ignore them silently. Without the flag, only the versions given are changed.
- `ONNXSequential` raises a `ValueError` naming both ends and both element types when an output and the input it
  feeds have different element types. The merge drops the declared input, so such a chain used to fail later with
  ONNX Runtime's `Type Error`, or, through an operator that takes any type such as `Cast`, to run although the second
  graph rejects that input on its own. Declared dimensions are not compared.
- A malformed `io_maps` raises a `BaseError` that says `io_maps` holds one list of `(output, input)` name pairs per
  joint, where it used to raise `IndexError` (an entry missing) or `ValueError: Output K00-m is not present in g1` (a
  flat list of pairs, the form the ONNX guide showed). The `io_maps` annotations of `ONNXSequential`,
  `ONNXSequential.combine` and `HFONNXComunnityModel` are now `list[list[tuple[str, str]]]`, and the guide's example
  is a list of lists. Entries past the last joint are still not used.
- `ONNXModule`, `ONNXSequential` and `kornia.onnx.load` take model paths as `os.PathLike` objects such as
  `pathlib.Path`, which used to raise `ValueError: Invalid argument type`. An argument of another type now raises
  `TypeError` instead of `ValueError`.
- Calling `ONNXModule` or `ONNXSequential` takes torch tensors on the CPU as well as numpy arrays, and still returns
  numpy arrays; a tensor used to raise ONNX Runtime's `RuntimeError: Input must be a list of dictionaries or a single
  numpy array`. A tensor on another device raises a `ValueError` that says to move it to the CPU.
- `ONNXLoader.load_config` reads an existing local JSON file, which it used to report as not found ("You may set
  `download=True`"). An argument that is neither an HTTP(S) URL nor an existing file raises a `RuntimeError` that
  says so.
- The `ONNXModule` docstring documents its parameter `op`, one model, and `io_name_conversion` documents that it
  applies any mapping in place and returns the same model.
