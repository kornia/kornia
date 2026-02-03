
<b>Pattern 1: Use Kornia’s KORNIA_CHECK / KORNIA_CHECK_SHAPE / check_* APIs for input validation instead of ad-hoc isinstance/shape checks, to keep error types/messages consistent across the codebase and tests.
</b>

Example code before:
```
def fn(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.ndim != 4:
        raise ValueError("Expected BCHW")
    return x
```

Example code after:
```
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

def fn(x):
    KORNIA_CHECK_IS_TENSOR(x)
    KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])
    return x
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3470#discussion_r2678895498
- https://github.com/kornia/kornia/pull/3482#discussion_r2681802420
- https://github.com/kornia/kornia/pull/3237#discussion_r2262340114
</details>


___

<b>Pattern 2: Avoid redundant or double-validation checks (e.g., tensor-type checks that are implicitly guaranteed by shape checks), and keep only the minimal set of checks needed for correctness and legacy behavior.
</b>

Example code before:
```
def fn(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("not a tensor")
    KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])
    return x
```

Example code after:
```
def fn(x):
    KORNIA_CHECK_SHAPE(x, ["B", "C", "H", "W"])  # implicitly ensures tensor-like shape semantics
    return x
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3482#discussion_r2685674255
- https://github.com/kornia/kornia/pull/3156#discussion_r2023847635
</details>


___

<b>Pattern 3: Keep tests maintainable by reducing duplication: prefer pytest parametrization and shared test helpers/BaseTester methods, and consolidate new tests into existing ones when they validate the same behavior.
</b>

Example code before:
```
def test_case1(): ...
def test_case2(): ...
def test_case3(): ...
```

Example code after:
```
@pytest.mark.parametrize("case", [case1, case2, case3])
def test_cases(case):
    ...
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3478#discussion_r2681874467
- https://github.com/kornia/kornia/pull/3330#discussion_r2477258202
- https://github.com/kornia/kornia/pull/3480#discussion_r2681840955
- https://github.com/kornia/kornia/pull/3407#discussion_r2649834312
</details>


___

<b>Pattern 4: When adding a new method/feature, reuse existing library utilities (conversions, IO helpers, etc.) rather than reimplementing equivalent logic, to ensure consistent behavior and reduce maintenance burden.
</b>

Example code before:
```
def to_gray(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b
```

Example code after:
```
from kornia.color import rgb_to_grayscale

def to_gray(rgb):
    return rgb_to_grayscale(rgb)
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3217#discussion_r2250860535
- https://github.com/kornia/kornia/pull/3407#discussion_r2649834445
</details>


___

<b>Pattern 5: Prefer minimal, “clean” source: remove commented-out code, irrelevant/dated comments, and avoid referencing issues inside code comments (use PR/issue tracker instead); keep docs/source formatting intentional (e.g., blank lines where required).
</b>

Example code before:
```
# TODO(issue #1234): remove this later
# old_impl = ...
do_work()
```

Example code after:
```
do_work()
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3466#discussion_r2668177156
- https://github.com/kornia/kornia/pull/3424#discussion_r2657066266
- https://github.com/kornia/kornia/pull/3209#discussion_r2244904526
- https://github.com/kornia/kornia/pull/3509#discussion_r2693843649
</details>


___

<b>Pattern 6: For performance-related changes, avoid repeated expensive computations inside hot paths (e.g., repeated reductions/sums/logs); compute once, cache constants on the module, and reuse intermediate results.
</b>

Example code before:
```
def forward(self, x):
    if weights.sum() == 0:
        weights = torch.ones_like(weights) / weights.numel()
    y = x * self.scale.clamp(0, math.log(self.scale_max)).exp()
    return y
```

Example code after:
```
def __init__(self, scale_max: float):
    super().__init__()
    self.scale_max_log = math.log(scale_max)

def forward(self, x):
    w_sum = weights.sum()
    if w_sum == 0:
        weights = torch.ones_like(weights) / weights.numel()
    y = x * self.scale.clamp(0, self.scale_max_log).exp()
    return y
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3424#discussion_r2657066917
- https://github.com/kornia/kornia/pull/3480#discussion_r2681909888
- https://github.com/kornia/kornia/pull/3156#discussion_r2024485622
</details>


___

<b>Pattern 7: When introducing device/dtype/layout-specific branches or forcing contiguous/copies, justify it with benchmarks and add a short comment explaining why the branch/copy is necessary and what might change in future (to avoid “mysterious” performance hacks).
</b>

Example code before:
```
x = x.contiguous()  # no explanation
if x.device.type == "cpu":
    return slow_path(x)
return fast_path(x)
```

Example code after:
```
# NOTE: CPU uses einsum since conv2d regresses on CPU for this op; GPU uses conv2d for speed.
# If PyTorch performance changes, revisit this dispatch.
if x.device.type == "cpu":
    return einsum_path(x)
return conv_path(x)
```

<details><summary>Examples for relevant past discussions:</summary>

- https://github.com/kornia/kornia/pull/3482#discussion_r2681811927
- https://github.com/kornia/kornia/pull/3482#discussion_r2685682657
</details>


___
