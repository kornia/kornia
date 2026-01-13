# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from functools import wraps
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Callable, Dict, List
import imageio
from tqdm import tqdm


def async_call_func(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        # Use run_in_executor to run the blocking function in a separate thread
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


slice_func = lambda chunk_index, chunk_dim, chunk_size: [slice(None)] * chunk_dim + [
    slice(chunk_index, chunk_index + chunk_size)
]


def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


def _save_image_impl(save_img, save_path):
    """Common implementation for saving images synchronously or asynchronously"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, save_img)


@async_call
def save_image_async(save_img, save_path):
    """Save image asynchronously"""
    _save_image_impl(save_img, save_path)


def save_image(save_img, save_path):
    """Save image synchronously"""
    _save_image_impl(save_img, save_path)


def parallel_execution(
    *args,
    action: Callable,
    num_processes=32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    **kwargs,
):
    # Partially copy from EasyVolumetricVideo (parallel_execution)
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved.
    # `*args` packs all positional arguments passed to the function into a tuple
    args = list(args)

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [
            (arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args
        ]
        # TODO: Support all types of iterable
        action_kwargs = {
            key: (
                kwargs[key][i]
                if isinstance(kwargs[key], list) and len(kwargs[key]) == length
                else kwargs[key]
            )
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:
        # Create ThreadPool
        pool = ThreadPool(processes=num_processes)

        # Spawn threads
        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)

        # Join threads and get return values
        if not async_return:
            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())  # will sync the corresponding thread
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
