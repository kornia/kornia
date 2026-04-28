# CUDA Graph Capture Harness

This harness attempts `torch.cuda.graph` capture on every public concrete
kornia.augmentation transform at a fixed input shape, and reports per-transform
capture pass/fail plus eager-vs-replay timing.

## What it measures

CUDA Graph capture fails whenever a PyTorch operation triggers a host sync
(e.g. `.item()`, shape-dependent branches, CPU-side RNG). A passing capture is
the binary proof that a transform is graph-safe. The replay speedup measures
how much launch-overhead elimination is worth.

## How to run

```bash
# Full benchmark (writes results.jsonl)
python -m benchmarks.cuda_graph.per_transform --output benchmarks/cuda_graph/results.jsonl

# Custom batch/resolution
python -m benchmarks.cuda_graph.per_transform --batch 4 --resolution 256

# Exits 0 if all transforms pass or CUDA unavailable; exits 1 if any FAILED
```

## How to update the leaderboard

```bash
python -m benchmarks.cuda_graph.leaderboard \
    --input benchmarks/cuda_graph/results.jsonl \
    --output benchmarks/cuda_graph/leaderboard.md
```

## CI gate

`benchmarks/cuda_graph/test_capture_contract.py` is the pytest entry point.
It is parametrized over every discovered transform. Any FAILED transform causes
the test to fail, making this a hard CI gate.

```bash
pytest benchmarks/cuda_graph/test_capture_contract.py -v
```

The test is automatically skipped when CUDA is unavailable.

## Pass/fail contract (RFC §3.4)

Capture pass/fail is the binary contract proving no host syncs. The post-PR-G1
(GPU-side RNG landing) target is **82/82 OK**. Until then, transforms using
CPU-side RNG or `.item()` calls will show as FAILED — these are expected and
tracked.
