# to install:
# pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117  # noqa: E501

from itertools import product

import cv2
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark

import kornia

torch_dynamo_optimize = dynamo.optimize("inductor")

op = kornia.color.rgb_to_grayscale
op_dyn = torch_dynamo_optimize(op)

# simulate batch as sequential op
# NOTE: we include the data transfer because eventually as op included in the pipeline


def op_encv(x):
    if len(x.shape) == 3:
        x = x[None]
    for i in range(x.shape[0]):
        x_np = kornia.tensor_to_image(x[i].mul_(255.0).byte())
        cv2.cvtColor(x_np, cv2.COLOR_RGB2GRAY)


# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [None] + [2**x for x in range(1, 2)]
resolution = [2**x for x in range(5, 10)]
threads = [1, 4]

for b, n in product(sizes, resolution):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched grayscale'
    sub_label = f'[{b}, {n}]'
    x = torch.ones((3, n, n))
    if b is not None:
        x = x[None].repeat(b, 1, 1, 1)
    for num_threads in threads:
        results.append(
            benchmark.Timer(
                stmt='op(image)',
                setup='from __main__ import op',
                globals={'image': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='to_gray_eager_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op(image)',
                setup='from __main__ import op',
                globals={'image': x.cuda()},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='to_gray_eager_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(image)',
                setup='from __main__ import op_dyn',
                globals={'image': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='to_gray_dyn_cpu',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_dyn(image)',
                setup='from __main__ import op_dyn',
                globals={'image': x.cuda()},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='to_gray_dyn_cuda',
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt='op_encv(image)',
                setup='from __main__ import op_encv',
                globals={'image': x},
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description='to_gray_opencv',
            ).blocked_autorange(min_run_time=1)
        )

compare = benchmark.Compare(results)
compare.print()
