For empty batches, RandomCrop and Resize inverse operations now return the recorded original height and width
rather than retaining transformed dimensions. This change preserves channel count, dtype, and device. Empty
RandomCrop forward_input_shape now records unpadded input dimensions rather than the padded canvas. Regenerate
previously saved empty RandomCrop parameters before inverse replay because they lack the original padding history.
Nonempty-batch metadata is unchanged. Part of #4429.

In multi-stage empty crop and resize pipelines, each intermediate inverse now returns the actual input size
recorded for that stage.
