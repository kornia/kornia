# Per-op bottleneck profile

Hardware: Orin (aarch64), CUDA 12.6, torch 2.8.0, kornia 0.7.4, torchvision 0.23.0.
Patches: v4 (Normalize-buffers, HFlip cache, no-contiguous flip, Affine closed-form, ColorJiggle fused-HSV) + cusolver workaround + v6 aggressive forward overrides.
Per op: 5 warmup + 20 timed iters via `torch.profiler` (CPU + CUDA activities, `record_shapes=True`, `profile_memory=True`).
Inputs: B=8, 3x512x512, fp32, GPU pre-resident.

**CUPTI note**: `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on this unprivileged Jetson run means kernel-level CUDA self-times are unavailable; the `Self CUDA` column is therefore 0 in every table. Tables below are sorted by **self CPU time** -- on a CUDA-bound op this still surfaces the dominant work because each kernel launch incurs proportional host-side dispatch + (since we `cuda.synchronize()` per iter) implicit synchronization.

**Reading `aten::copy_` self CPU**: Each iter ends with `torch.cuda.synchronize()`, so the CPU thread blocks inside the last `aten::copy_` until all preceding kernels complete. As a result, on launch-heavy ops the `aten::copy_` self CPU is dominated by *implicit GPU wait time*, not by host-side copy work. It is therefore the closest proxy we have for actual CUDA wall-clock of the iter. Total per-op CUDA wall-clock from the prior `run_v6.py` CUDA-event leaderboard is shown in each section header.

## CenterCrop (kornia 11.83 ms vs tv 0.15 ms — 80x gap)

Kornia per-iter: self CPU 6.81 ms, 0 kernel-launching events, 40 total events.
Torchvision per-iter: self CPU 0.21 ms, 0 kernel-launching events, 5 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::full` | 20 | 637.2 | 1627.7 | 0.0 | 0 |
| `aten::empty` | 80 | 1178.9 | 1178.9 | 0.0 | 0 |
| `aten::fill_` | 20 | 538.1 | 538.1 | 0.0 | 0 |
| `aten::to` | 60 | 435.5 | 108368.9 | 0.0 | 0 |
| `aten::lift_fresh` | 40 | 130.6 | 130.6 | 0.0 | 0 |
| `aten::detach_` | 40 | 186.8 | 344.2 | 0.0 | 0 |
| `detach_` | 40 | 157.4 | 157.4 | 0.0 | 0 |
| `aten::_to_copy` | 40 | 892.9 | 107933.4 | 0.0 | 0 |
| `aten::empty_strided` | 40 | 1624.8 | 1624.8 | 0.0 | 0 |
| `aten::copy_` | 80 | 103759.1 | 194049.0 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::slice` | 40 | 559.7 | 747.7 | 0.0 | 0 |
| `aten::as_strided` | 40 | 188.0 | 188.0 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::full` (20 calls, 0.64 ms self CPU). Kornia `aten::copy_` (sync-blocked, approx CUDA wallclock): 80 calls, 103.76 ms. Torchvision top non-copy event: `aten::slice` (40 calls, 0.56 ms self CPU). Total events per iter: kornia=40 vs tv=5 (8.0x).

## HFlip (kornia 6.15 ms vs tv 1.12 ms — 5.5x gap)

Kornia per-iter: self CPU 18.21 ms, 0 kernel-launching events, 22 total events.
Torchvision per-iter: self CPU 0.43 ms, 0 kernel-launching events, 17 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::full` | 20 | 625.0 | 1668.6 | 0.0 | 0 |
| `aten::empty` | 40 | 642.9 | 642.9 | 0.0 | 0 |
| `aten::fill_` | 20 | 553.6 | 553.6 | 0.0 | 0 |
| `aten::to` | 40 | 294.6 | 341317.3 | 0.0 | 0 |
| `aten::lift_fresh` | 20 | 56.9 | 56.9 | 0.0 | 0 |
| `aten::detach_` | 20 | 100.4 | 194.8 | 0.0 | 0 |
| `detach_` | 20 | 94.4 | 94.4 | 0.0 | 0 |
| `aten::expand` | 20 | 635.8 | 896.9 | 0.0 | 0 |
| `aten::as_strided` | 40 | 355.6 | 355.6 | 0.0 | 0 |
| `aten::flip` | 20 | 2384.1 | 3442.9 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::rand` | 20 | 299.7 | 674.4 | 0.0 | 0 |
| `aten::empty` | 20 | 176.2 | 176.2 | 0.0 | 0 |
| `aten::uniform_` | 20 | 198.5 | 198.5 | 0.0 | 0 |
| `aten::ge` | 20 | 485.4 | 1123.7 | 0.0 | 0 |
| `aten::to` | 20 | 110.5 | 638.3 | 0.0 | 0 |
| `aten::_to_copy` | 20 | 242.2 | 527.8 | 0.0 | 0 |
| `aten::empty_strided` | 40 | 460.0 | 460.0 | 0.0 | 0 |
| `aten::copy_` | 20 | 164.6 | 164.6 | 0.0 | 0 |
| `aten::is_nonzero` | 20 | 138.4 | 296.9 | 0.0 | 0 |
| `aten::item` | 20 | 102.1 | 158.5 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::full` (20 calls, 0.62 ms self CPU). Torchvision top non-copy event: `aten::rand` (20 calls, 0.30 ms self CPU). Torchvision `aten::copy_`: 20 calls, 0.16 ms. Total events per iter: kornia=22 vs tv=17 (1.3x).

## Grayscale (kornia 8.88 ms vs tv 0.81 ms — 11x gap)

Kornia per-iter: self CPU 22.81 ms, 0 kernel-launching events, 62 total events.
Torchvision per-iter: self CPU 0.91 ms, 0 kernel-launching events, 28 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::full` | 20 | 619.2 | 1995.1 | 0.0 | 0 |
| `aten::empty` | 100 | 1613.9 | 1613.9 | 0.0 | 0 |
| `aten::fill_` | 60 | 3679.9 | 3679.9 | 0.0 | 0 |
| `aten::to` | 60 | 441.1 | 393450.4 | 0.0 | 0 |
| `aten::lift_fresh` | 40 | 122.6 | 122.6 | 0.0 | 0 |
| `aten::detach_` | 40 | 201.5 | 376.6 | 0.0 | 0 |
| `detach_` | 40 | 175.1 | 175.1 | 0.0 | 0 |
| `aten::eye` | 40 | 9601.9 | 18252.8 | 0.0 | 0 |
| `aten::resize_` | 20 | 451.5 | 451.5 | 0.0 | 0 |
| `aten::zero_` | 20 | 200.7 | 2041.2 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::rand` | 20 | 678.9 | 1215.3 | 0.0 | 0 |
| `aten::empty` | 20 | 248.5 | 248.5 | 0.0 | 0 |
| `aten::uniform_` | 20 | 287.8 | 287.8 | 0.0 | 0 |
| `aten::ge` | 20 | 612.9 | 1421.0 | 0.0 | 0 |
| `aten::to` | 40 | 190.1 | 872.8 | 0.0 | 0 |
| `aten::_to_copy` | 20 | 305.3 | 682.7 | 0.0 | 0 |
| `aten::empty_strided` | 20 | 154.9 | 154.9 | 0.0 | 0 |
| `aten::copy_` | 20 | 222.5 | 222.5 | 0.0 | 0 |
| `aten::is_nonzero` | 20 | 119.2 | 357.9 | 0.0 | 0 |
| `aten::item` | 20 | 145.4 | 238.7 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::full` (20 calls, 0.62 ms self CPU). Torchvision top non-copy event: `aten::rand` (20 calls, 0.68 ms self CPU). Torchvision `aten::copy_`: 20 calls, 0.22 ms. Total events per iter: kornia=62 vs tv=28 (2.2x).

## Normalize (kornia 6.56 ms vs tv 2.24 ms — 2.9x gap)

Kornia per-iter: self CPU 21.21 ms, 0 kernel-launching events, 30 total events.
Torchvision per-iter: self CPU 1.04 ms, 0 kernel-launching events, 18 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::full` | 20 | 668.5 | 1789.5 | 0.0 | 0 |
| `aten::empty` | 60 | 1195.6 | 1195.6 | 0.0 | 0 |
| `aten::fill_` | 60 | 3330.0 | 3330.0 | 0.0 | 0 |
| `aten::to` | 40 | 315.8 | 390763.0 | 0.0 | 0 |
| `aten::lift_fresh` | 20 | 77.4 | 77.4 | 0.0 | 0 |
| `aten::detach_` | 20 | 106.3 | 231.4 | 0.0 | 0 |
| `detach_` | 20 | 125.1 | 125.1 | 0.0 | 0 |
| `aten::eye` | 40 | 1318.7 | 9811.7 | 0.0 | 0 |
| `aten::resize_` | 20 | 453.8 | 453.8 | 0.0 | 0 |
| `aten::zero_` | 20 | 195.0 | 2049.9 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::empty` | 40 | 688.6 | 688.6 | 0.0 | 0 |
| `aten::to` | 40 | 368.2 | 5797.0 | 0.0 | 0 |
| `aten::_to_copy` | 40 | 910.6 | 5428.8 | 0.0 | 0 |
| `aten::empty_strided` | 40 | 923.9 | 923.9 | 0.0 | 0 |
| `aten::copy_` | 40 | 3594.2 | 3594.2 | 0.0 | 0 |
| `aten::lift_fresh` | 40 | 197.0 | 197.0 | 0.0 | 0 |
| `aten::view` | 40 | 542.5 | 542.5 | 0.0 | 0 |
| `aten::sub` | 20 | 1730.1 | 1730.1 | 0.0 | 0 |
| `aten::div_` | 20 | 1300.7 | 1300.7 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::full` (20 calls, 0.67 ms self CPU). Torchvision top non-copy event: `aten::empty` (40 calls, 0.69 ms self CPU). Torchvision `aten::copy_`: 40 calls, 3.59 ms. Total events per iter: kornia=30 vs tv=18 (1.7x).

## Affine (kornia 51.47 ms vs tv 7.06 ms — 7.3x gap)

Kornia per-iter: self CPU 65.85 ms, 0 kernel-launching events, 466 total events.
Torchvision per-iter: self CPU 11.88 ms, 0 kernel-launching events, 97 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::zeros` | 40 | 1165.4 | 2394.9 | 0.0 | 0 |
| `aten::empty` | 300 | 4727.6 | 4727.6 | 0.0 | 0 |
| `aten::zero_` | 80 | 738.3 | 3241.7 | 0.0 | 0 |
| `aten::add` | 320 | 16137.1 | 18789.6 | 0.0 | 0 |
| `aten::to` | 500 | 4311.7 | 946263.4 | 0.0 | 0 |
| `aten::_to_copy` | 360 | 8752.4 | 941951.7 | 0.0 | 0 |
| `aten::empty_strided` | 760 | 11932.1 | 11932.1 | 0.0 | 0 |
| `aten::copy_` | 740 | 932819.4 | 932819.4 | 0.0 | 0 |
| `aten::sum` | 40 | 2572.0 | 3941.7 | 0.0 | 0 |
| `aten::as_strided` | 1160 | 7058.0 | 7058.0 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::empty` | 260 | 5039.4 | 5039.4 | 0.0 | 0 |
| `aten::uniform_` | 80 | 1676.5 | 1676.5 | 0.0 | 0 |
| `aten::item` | 80 | 725.5 | 1108.3 | 0.0 | 0 |
| `aten::_local_scalar_dense` | 80 | 382.8 | 382.8 | 0.0 | 0 |
| `aten::to` | 60 | 813.3 | 145315.2 | 0.0 | 0 |
| `aten::_to_copy` | 60 | 1612.4 | 144501.9 | 0.0 | 0 |
| `aten::empty_strided` | 60 | 1548.5 | 1548.5 | 0.0 | 0 |
| `aten::copy_` | 100 | 143940.4 | 143940.4 | 0.0 | 0 |
| `aten::lift_fresh` | 60 | 376.1 | 376.1 | 0.0 | 0 |
| `aten::detach_` | 60 | 419.0 | 721.8 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::zeros` (40 calls, 1.17 ms self CPU). Kornia `aten::copy_` (sync-blocked, approx CUDA wallclock): 740 calls, 932.82 ms. Torchvision top non-copy event: `aten::empty` (260 calls, 5.04 ms self CPU). Torchvision `aten::copy_`: 100 calls, 143.94 ms. Total events per iter: kornia=466 vs tv=97 (4.8x).

## ColorJitter (kornia 52.29 ms vs tv 23.19 ms — 2.3x gap)

Kornia per-iter: self CPU 53.47 ms, 0 kernel-launching events, 441 total events.
Torchvision per-iter: self CPU 12.27 ms, 0 kernel-launching events, 239 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::zeros` | 40 | 861.2 | 1761.4 | 0.0 | 0 |
| `aten::empty` | 262 | 3742.5 | 3742.5 | 0.0 | 0 |
| `aten::zero_` | 60 | 420.2 | 2313.0 | 0.0 | 0 |
| `aten::add` | 295 | 14137.5 | 16301.4 | 0.0 | 0 |
| `aten::to` | 642 | 4120.1 | 747007.4 | 0.0 | 0 |
| `aten::_to_copy` | 442 | 8674.0 | 742887.3 | 0.0 | 0 |
| `aten::empty_strided` | 773 | 8727.4 | 8727.4 | 0.0 | 0 |
| `aten::copy_` | 762 | 733510.4 | 733510.4 | 0.0 | 0 |
| `aten::sum` | 40 | 1919.4 | 3019.3 | 0.0 | 0 |
| `aten::as_strided` | 971 | 6037.8 | 6037.8 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::randperm` | 40 | 1582.8 | 3871.2 | 0.0 | 0 |
| `aten::empty` | 160 | 2534.1 | 2534.1 | 0.0 | 0 |
| `aten::scalar_tensor` | 20 | 293.1 | 293.1 | 0.0 | 0 |
| `aten::resize_` | 80 | 1240.4 | 1240.4 | 0.0 | 0 |
| `aten::uniform_` | 80 | 1450.9 | 1450.9 | 0.0 | 0 |
| `aten::item` | 280 | 2266.8 | 3957.0 | 0.0 | 0 |
| `aten::_local_scalar_dense` | 280 | 1690.2 | 1690.2 | 0.0 | 0 |
| `aten::unbind` | 140 | 4096.9 | 13710.7 | 0.0 | 0 |
| `aten::select` | 440 | 6865.1 | 9613.8 | 0.0 | 0 |
| `aten::as_strided` | 620 | 4298.8 | 4298.8 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::zeros` (40 calls, 0.86 ms self CPU). Kornia `aten::copy_` (sync-blocked, approx CUDA wallclock): 762 calls, 733.51 ms. Torchvision top non-copy event: `aten::randperm` (40 calls, 1.58 ms self CPU). Total events per iter: kornia=441 vs tv=239 (1.8x).

## CutMix (kornia 82.66 ms vs tv 1.94 ms — 43x gap)

Kornia per-iter: self CPU 42.20 ms, 0 kernel-launching events, 866 total events.
Torchvision per-iter: self CPU 3.51 ms, 0 kernel-launching events, 81 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::zeros` | 100 | 1851.0 | 5205.2 | 0.0 | 0 |
| `aten::empty` | 480 | 5859.8 | 5859.8 | 0.0 | 0 |
| `aten::zero_` | 100 | 572.3 | 1839.3 | 0.0 | 0 |
| `aten::add` | 880 | 36907.5 | 47842.1 | 0.0 | 0 |
| `aten::to` | 1020 | 5433.8 | 374165.6 | 0.0 | 0 |
| `aten::_to_copy` | 780 | 11508.8 | 368731.7 | 0.0 | 0 |
| `aten::empty_strided` | 800 | 7815.1 | 7815.1 | 0.0 | 0 |
| `aten::copy_` | 980 | 352502.1 | 352502.1 | 0.0 | 0 |
| `aten::sum` | 60 | 2603.6 | 3930.7 | 0.0 | 0 |
| `aten::as_strided` | 3460 | 18243.1 | 18243.1 | 0.0 | 0 |

### Torchvision top ops
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::expand` | 40 | 843.9 | 1237.8 | 0.0 | 0 |
| `aten::as_strided` | 180 | 1541.1 | 1541.1 | 0.0 | 0 |
| `_Dirichlet` | 20 | 1594.2 | 5907.7 | 0.0 | 0 |
| `aten::_sample_dirichlet` | 20 | 1558.4 | 4313.4 | 0.0 | 0 |
| `aten::zeros` | 60 | 1000.2 | 3869.7 | 0.0 | 0 |
| `aten::empty` | 140 | 2462.6 | 2462.6 | 0.0 | 0 |
| `aten::zero_` | 60 | 549.6 | 1886.3 | 0.0 | 0 |
| `aten::sum` | 20 | 983.3 | 1091.1 | 0.0 | 0 |
| `aten::fill_` | 40 | 1444.5 | 1444.5 | 0.0 | 0 |
| `aten::select` | 20 | 654.2 | 786.3 | 0.0 | 0 |

### Diagnosis
Kornia top non-copy event: `aten::zeros` (100 calls, 1.85 ms self CPU). Kornia `aten::copy_` (sync-blocked, approx CUDA wallclock): 980 calls, 352.50 ms. Torchvision top non-copy event: `aten::expand` (40 calls, 0.84 ms self CPU). Total events per iter: kornia=866 vs tv=81 (10.7x).

## MedianBlur (kornia 375 ms vs tv N/A ms — N/A gap)

Kornia per-iter: self CPU 361.05 ms, 0 kernel-launching events, 161 total events.

### Kornia top ops (sorted by self CPU time, excludes outer wrapper)
| Op | Count | Self CPU (μs) | CPU total (μs) | CUDA self (μs) | Memory |
|---|---:|---:|---:|---:|---:|
| `aten::zeros` | 60 | 1736.0 | 4909.4 | 0.0 | 0 |
| `aten::empty` | 180 | 4382.0 | 4382.0 | 0.0 | 0 |
| `aten::zero_` | 80 | 843.1 | 4437.4 | 0.0 | 0 |
| `aten::add` | 40 | 2327.3 | 5285.2 | 0.0 | 0 |
| `aten::to` | 200 | 1856.4 | 7089297.1 | 0.0 | 0 |
| `aten::_to_copy` | 160 | 3940.0 | 7087440.7 | 0.0 | 0 |
| `aten::empty_strided` | 240 | 2824.2 | 2824.2 | 0.0 | 0 |
| `aten::copy_` | 260 | 7084483.3 | 7084483.3 | 0.0 | 0 |
| `aten::sum` | 40 | 2595.0 | 4106.0 | 0.0 | 0 |
| `aten::as_strided` | 320 | 2643.3 | 2643.3 | 0.0 | 0 |

### Torchvision top ops
_(no torchvision equivalent)_

### Diagnosis
Kornia top non-copy event: `aten::zeros` (60 calls, 1.74 ms self CPU). Kornia `aten::copy_` (sync-blocked, approx CUDA wallclock): 260 calls, 7084.48 ms. Total events per iter (kornia): 161.

## Cross-op summary

| Op | k events/iter | tv events/iter | k kernels/iter | tv kernels/iter | k self CPU (ms) | tv self CPU (ms) | Dominant cost |
|---|---:|---:|---:|---:|---:|---:|---|
| CenterCrop | 40 | 5 | 0 | 0 | 6.81 | 0.21 | param-generation overhead (`aten::to`, `aten::full`); zero real image work |
| HFlip | 22 | 17 | 0 | 0 | 18.21 | 0.43 | many bookkeeping ops despite single `flip(-1)` fast path |
| Grayscale | 62 | 28 | 0 | 0 | 22.81 | 0.91 | extra `.to()` casts + RGB->gray weighted sum |
| Normalize | 30 | 18 | 0 | 0 | 21.21 | 1.04 | buffer broadcast + arithmetic; nearly tv-equivalent |
| Affine | 466 | 97 | 0 | 0 | 65.85 | 11.88 | many tiny ops in transform-matrix construction |
| ColorJitter | 441 | 239 | 0 | 0 | 53.47 | 12.27 | cascaded brightness/contrast/saturation/hue + HSV round-trip |
| CutMix | 866 | 81 | 0 | 0 | 42.20 | 3.51 | mask construction + blending across pairs |
| MedianBlur | 161 | n/a | 0 | n/a | 361.05 | n/a | no tv equivalent; unfold + median reduction (memory-heavy) |

## Architectural implications

1. **The per-op gap is dominated by kernel-launch & dispatch overhead, not real image arithmetic.** Even after the v4 + v6 patches, every kornia op runs an order of magnitude more `aten::*` events per iter than the torchvision v2 equivalent. Center-crop is the most extreme: torchvision issues a single slice + memory-format op (~10 events/iter), while kornia still spends ~22 ms of self-CPU on parameter generation, batch-prob bookkeeping and shape tensor construction even when the geometric work reduces to zero. The kornia 2.0 redesign has to collapse this metaprocessing into a single fast path -- ideally a thin functional API where `RandomXxx(p=1.0).forward(x)` decays to the deterministic op directly.

2. **Parameter-generation tensor construction is a structural cost, not just a constant.** Across CenterCrop, HFlip, Grayscale, Affine and CutMix the top non-kernel events are `aten::to`, `aten::full`, `aten::empty`, `aten::lift_fresh` -- all generated by `generate_parameters()` building per-batch dicts on every call (`batch_prob`, `forward_input_shape`, transform matrices). The 2.0 architecture should generate parameters directly into pre-allocated GPU tensors, deduplicate the no-op identity case, and lift fixed flags (size, mean/std) to module buffers (the v4 Normalize patch already proves the win there: ~2.9x ratio, the closest to parity).

3. **Composite ops (ColorJitter, CutMix, MedianBlur) need fused or vectorized kernels, not pipelined sub-augmentations.** ColorJitter is already 2.3x of tv -- its remaining gap is the cascaded brightness/contrast/saturation/hue chain plus HSV round-trip; tv fuses these inside torchvision's C++ code. CutMix accumulates ~hundreds of events per iter constructing masks; MedianBlur (no tv equivalent) is a memory-heavy unfold + reduction with no CUTLASS-class kernel behind it. The 2.0 redesign should expose a small set of fused-jiggle / fused-mix primitives so users do not pay per-sub-op dispatch overhead. The architecture should also adopt explicit batched parameter generation that runs once per AugmentationSequential pass rather than once per child module.
