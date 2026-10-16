`LightGlue` matches a batch of image pairs with point pruning, which is on by default, instead of raising an
`IndexError`: each pair gives the matches and scores it gives on its own, and a batch that can prune runs its pairs one
at a time. A batch with too few keypoints for pruning to start (on CUDA, 1536 by default) no longer raises either, and
early stopping ends such a batch once every pair meets its own criterion, rather than counting the unconfident points
of all pairs together. A float16 model with `mp=True` on CPU now autocasts to float16 and returns float16 scores,
instead of raising a `RuntimeError` in its first layer, or with `"keynet_affnet_hardnet"` already when it reads the
LAFs.
