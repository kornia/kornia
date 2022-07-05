from itertools import product

import torch
import torch.utils.benchmark as benchmark

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = 'get_perspective_transform'
    sub_label = f'[{b}, {4}, {2}]'
    x = torch.rand((b, 4, 2))
    for num_threads in [1, 4, 16, 32]:
        results.append(benchmark.Timer(
            stmt='get_perspective_transform(x, x)',
            setup='from kornia.geometry import get_perspective_transform',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='get_perspective_transform',
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.print()
