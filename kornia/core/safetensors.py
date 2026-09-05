# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Read ``.safetensors`` checkpoints with nothing but :mod:`torch`.

The format is small enough to read directly -- an 8-byte little-endian length,
a JSON header of that length, then one contiguous byte buffer the header indexes
into -- so a checkpoint published in it costs kornia no dependency. See
https://github.com/huggingface/safetensors#format for the specification.
"""

from __future__ import annotations

import json
import mmap
import os
from typing import Any

import torch

__all__ = ["load_safetensors"]

_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
"""The dtype names this reader accepts, mapped to their torch equivalents.

The format also defines the unsigned widths above 8 bits and two 8-bit float
encodings. They are left out rather than guessed at: torch's ``uint16``/``uint32``
/``uint64`` support only a fraction of the operator surface, and the ``F8``
encodings need a scale that the format does not carry. A checkpoint using one is
rejected by name instead of being read as something else.
"""

_ITEMSIZES: dict[str, int] = {name: torch.empty(0, dtype=dtype).element_size() for name, dtype in _DTYPES.items()}
"""Bytes per element for each accepted dtype, taken from torch rather than written down."""

_HEADER_LEN_BYTES = 8
"""Width of the little-endian unsigned integer the file opens with."""

_MAX_HEADER_BYTES = 100_000_000
"""Ceiling on the declared header length, matching the reference implementation.

The length is read before anything is validated, so a corrupt or hostile file can
name a header of any size a 64-bit integer can hold. Without a bound, the read
below would be asked for that many bytes.
"""


def _parse_entry(path: str, name: str, entry: Any, data_len: int) -> tuple[torch.dtype, list[int], int, int]:
    """Validate one header entry and return its dtype, shape, offset and element count.

    Args:
        path: the file the entry came from, named in every error message so the
            caller has something to delete or re-download.
        name: the tensor's key in the header.
        entry: the header value for *name*, which a valid file states as a dict
            with ``dtype``, ``shape`` and ``data_offsets``.
        data_len: the size of the byte buffer the offsets index into.

    Returns:
        The torch dtype, the shape, the offset of the tensor's first byte
        relative to the start of the byte buffer, and its number of elements.

    Raises:
        ValueError: if the entry is malformed, names a dtype this reader does not
            accept, or describes a byte range that is outside the buffer or the
            wrong size for its shape and dtype.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: header entry {name!r} is {type(entry).__name__}, expected an object.")
    missing = [key for key in ("dtype", "shape", "data_offsets") if key not in entry]
    if missing:
        raise ValueError(f"{path}: header entry {name!r} is missing {', '.join(missing)}.")

    dtype_name = entry["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError(
            f"{path}: tensor {name!r} has dtype {dtype_name!r}, which this reader does not support. "
            f"Supported dtypes: {', '.join(sorted(_DTYPES))}."
        )
    dtype = _DTYPES[dtype_name]

    shape = entry["shape"]
    # ``bool`` is an ``int`` subclass and would pass a bare isinstance check, so
    # a shape of ``[True]`` must not read as ``[1]``.
    if not isinstance(shape, list) or any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape):
        raise ValueError(f"{path}: tensor {name!r} has shape {shape!r}, expected a list of non-negative integers.")

    offsets = entry["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(o, int) or o < 0 for o in offsets):
        raise ValueError(f"{path}: tensor {name!r} has data_offsets {offsets!r}, expected two non-negative integers.")
    start, end = offsets
    if start > end or end > data_len:
        raise ValueError(
            f"{path}: tensor {name!r} claims bytes [{start}, {end}) of a {data_len}-byte buffer, "
            f"which is not a range inside it."
        )

    numel = 1
    for dim in shape:
        numel *= dim
    expected = numel * _ITEMSIZES[dtype_name]
    if end - start != expected:
        raise ValueError(
            f"{path}: tensor {name!r} is {dtype_name}{shape}, which is {expected} bytes, "
            f"but its data_offsets span {end - start}."
        )
    return dtype, shape, start, numel


def load_safetensors(path: str | os.PathLike[str], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load a ``.safetensors`` checkpoint into a state dict.

    A pure-torch reader for the format described at
    https://github.com/huggingface/safetensors#format: eight bytes of
    little-endian header length, a JSON header of that length naming each
    tensor's dtype, shape and byte range, and one contiguous byte buffer those
    ranges index into. The optional ``__metadata__`` key is ignored.

    The file is memory-mapped rather than read whole, so a multi-gigabyte
    checkpoint is not held in memory twice. The mapping is private
    (:data:`mmap.ACCESS_COPY`) rather than read-only: :func:`torch.frombuffer`
    warns on a buffer it cannot write to, and a private mapping is writable
    without touching the file -- nothing here writes to it, so no page is ever
    copied. Every tensor is copied out of the mapping before it is returned, so
    the returned state dict owns its storage and the file is closed by the time
    this function returns.

    Args:
        path: the checkpoint to read.
        device: the device the returned tensors are placed on.

    Returns:
        The state dict, in the order the header lists it.

    Raises:
        ValueError: if the file is not a readable safetensors checkpoint --
            truncated, a header that is not JSON, an entry naming a dtype this
            reader does not accept, or a byte range that does not match the
            tensor it belongs to. Every message names the file.
        OSError: if the file cannot be opened or mapped.

    Example:
        >>> state_dict = load_safetensors("model.safetensors")  # doctest: +SKIP
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size < _HEADER_LEN_BYTES:
            raise ValueError(f"{path}: {size} bytes is too short to be a safetensors file.")
        header_len = int.from_bytes(f.read(_HEADER_LEN_BYTES), "little", signed=False)
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: the header declares {header_len} bytes, more than the {_MAX_HEADER_BYTES} cap.")
        data_start = _HEADER_LEN_BYTES + header_len
        if data_start > size:
            raise ValueError(
                f"{path}: the header declares {header_len} bytes but the file holds "
                f"{size - _HEADER_LEN_BYTES} after the length prefix."
            )
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}: the header is not valid JSON: {e}") from e
        if not isinstance(header, dict):
            raise ValueError(f"{path}: the header is a JSON {type(header).__name__}, expected an object.")

        entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
        data_len = size - data_start
        parsed = {name: _parse_entry(path, name, entry, data_len) for name, entry in entries.items()}

        state_dict: dict[str, torch.Tensor] = {}
        # ``ACCESS_COPY`` maps the whole file, so the offsets below are file
        # offsets: the byte buffer starts at ``data_start``.
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as buf:
            for name, (dtype, shape, start, numel) in parsed.items():
                if numel == 0:
                    # ``torch.frombuffer`` rejects a count of 0, and an empty
                    # tensor stores no bytes to point at anyway.
                    state_dict[name] = torch.empty(shape, dtype=dtype, device=device)
                    continue
                flat = torch.frombuffer(buf, dtype=dtype, count=numel, offset=data_start + start)
                # ``copy=True`` is what detaches the result from the mapping; the
                # tensor ``frombuffer`` returns is a view of it.
                state_dict[name] = flat.reshape(shape).to(device, copy=True)
    return state_dict
