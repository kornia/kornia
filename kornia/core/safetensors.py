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
import reprlib
from collections.abc import Iterable
from typing import Any, BinaryIO, NamedTuple

import torch

__all__ = ["check_safetensors", "load_safetensors"]


class _TensorEntry(NamedTuple):
    """One header entry, validated: what to read, from where, and how much."""

    dtype: torch.dtype
    shape: list[int]
    start: int
    end: int
    numel: int


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

_MAX_DIM = torch.iinfo(torch.int64).max
"""The largest dimension a torch tensor can have: its sizes are 64-bit signed."""

_MAX_HEADER_BYTES = 100_000_000
"""Ceiling on the declared header length, matching the reference implementation.

The length is read before anything is validated, so a corrupt or hostile file can
name a header of any size a 64-bit integer can hold. Without a bound, the read
below would be asked for that many bytes.
"""

_REPR = reprlib.Repr()
_REPR.maxlevel = 3
_REPR.maxlist = 8
_REPR.maxdict = 8
_REPR.maxstring = 200
_REPR.maxlong = 40
_REPR.maxother = 80


def _short(value: Any) -> str:
    """Return ``repr(value)``, cut short.

    Every value an error message quotes comes from the header, and the file
    decides how large it is: under the header cap, a ``dtype`` can be a list of
    ten million numbers, which quoted whole is a 30 MB message.
    """
    return _REPR.repr(value)


def _clip(text: str, limit: int = 200) -> str:
    """Cut *text* to *limit* characters; torch quotes every size in some errors."""
    return text if len(text) <= limit else text[:limit] + " ..."


def _bounded_product(factors: Iterable[int], limit: int) -> int | None:
    """Multiply positive *factors*, giving up once the product passes *limit*.

    Multiplying a long list of large factors all the way out is quadratic in
    their number, since every factor adds a word to the product: a header of
    100 000 dimensions of ``2**62`` takes about a minute. Here every
    multiplication starts from at most *limit*, so the cost is linear.

    Returns:
        The exact product when every factor was used, which can exceed *limit*
        by the last one; ``None`` when a partial product passed *limit* first.
    """
    product = 1
    for factor in factors:
        if product > limit:
            return None
        product *= factor
    return product


def _above_one(shape: list[int]) -> Iterable[int]:
    """The dimensions of *shape* above 1, the only ones that change a product, found at C speed.

    Every one of them at least doubles the product, so a product bounded below
    ``2**63`` multiplies at most 64 of them before it stops, however many
    dimensions the header lists.
    """
    return filter((1).__lt__, shape)


def _parse_entry(path: str, name: str, entry: Any, data_len: int) -> _TensorEntry:
    """Validate one header entry and return what is needed to read it.

    Args:
        path: the file the entry came from, named in every error message so the
            caller has something to delete or re-download.
        name: the tensor's key in the header.
        entry: the header value for *name*, which a valid file states as a dict
            with ``dtype``, ``shape`` and ``data_offsets``.
        data_len: the size of the byte buffer the offsets index into.

    Returns:
        The entry's dtype, shape, byte range relative to the start of the byte
        buffer, and number of elements.

    Raises:
        ValueError: if the entry is malformed, names a dtype this reader does not
            accept, has a shape torch cannot build, or describes a byte range that
            is outside the buffer or the wrong size for its shape and dtype.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: header entry {_short(name)} is {type(entry).__name__}, expected an object.")
    missing = [key for key in ("dtype", "shape", "data_offsets") if key not in entry]
    if missing:
        raise ValueError(f"{path}: header entry {_short(name)} is missing {', '.join(missing)}.")

    dtype_name = entry["dtype"]
    # A JSON list or object is unhashable, so the membership test alone would
    # raise ``TypeError`` rather than reject it.
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ValueError(
            f"{path}: tensor {_short(name)} has dtype {_short(dtype_name)}, which this reader does not support. "
            f"Supported dtypes: {', '.join(sorted(_DTYPES))}."
        )
    dtype = _DTYPES[dtype_name]

    shape = entry["shape"]
    # ``bool`` is an ``int`` subclass and would pass a bare isinstance check, so
    # a shape of ``[True]`` must not read as ``[1]``.
    if not isinstance(shape, list) or any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape):
        raise ValueError(
            f"{path}: tensor {_short(name)} has shape {_short(shape)}, expected a list of non-negative integers."
        )
    if max(shape, default=0) > _MAX_DIM:
        raise ValueError(
            f"{path}: tensor {_short(name)} has shape {_short(shape)}, whose dimensions must be at most 2**63 - 1, "
            f"the largest size a torch tensor can have."
        )

    offsets = entry["data_offsets"]
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(o, int) or isinstance(o, bool) or o < 0 for o in offsets)
    ):
        raise ValueError(
            f"{path}: tensor {_short(name)} has data_offsets {_short(offsets)}, expected two non-negative integers."
        )
    start, end = offsets
    if start > end or end > data_len:
        raise ValueError(
            f"{path}: tensor {_short(name)} claims bytes [{_short(start)}, {_short(end)}) of a {data_len}-byte buffer, "
            f"which is not a range inside it."
        )

    itemsize = _ITEMSIZES[dtype_name]
    # A count past what the buffer holds cannot match any range inside it, so
    # the product stops there instead of multiplying every dimension out.
    numel = 0 if 0 in shape else _bounded_product(_above_one(shape), data_len // itemsize)
    if numel is None or numel * itemsize != end - start:
        size = f"more than the {data_len}-byte buffer holds" if numel is None else f"{numel * itemsize} bytes"
        raise ValueError(
            f"{path}: tensor {_short(name)} is {dtype_name}{_short(shape)}, which is {size}, "
            f"but its data_offsets span {end - start}."
        )
    if numel == 0:
        # An empty tensor spans no bytes, so the check above bounds none of its
        # other dimensions, and torch still derives contiguous strides and a
        # storage size from them in 64-bit arithmetic that can overflow. While
        # the product of ``max(d, 1)`` stays below 2**63, neither can: every
        # stride is a product of a suffix of those factors, and every partial
        # product of the sizes is at most their product, or 0 from the first 0.
        # Factors of 0 and 1 contribute 1, so only the larger ones are multiplied.
        # Past that bound, building the tensor on the meta device runs the same
        # checks as the ``torch.empty`` in :func:`load_safetensors` does on the
        # CPU, without allocating its storage, so a shape accepted here is one
        # the load can build there.
        extent = _bounded_product(_above_one(shape), _MAX_DIM)
        if extent is None or extent > _MAX_DIM:
            try:
                torch.empty(shape, dtype=dtype, device="meta")
            except RuntimeError as e:
                raise ValueError(
                    f"{path}: tensor {_short(name)} has shape {_short(shape)}, which torch cannot build: "
                    f"{_clip(str(e))}"
                ) from e
    return _TensorEntry(dtype, shape, start, end, numel)


def _check_the_buffer_is_covered_once(path: str, parsed: dict[str, _TensorEntry], data_len: int) -> None:
    """Reject a byte buffer the entries do not tile exactly.

    The format requires the buffer to be entirely indexed and free of holes,
    which the per-entry checks do not give: each one only asks whether its own
    range fits. Two entries claiming ``[0, 8]`` and ``[4, 8]`` both fit, and
    would be read as two tensors sharing four bytes -- a file that no writer
    produces and that no reader should quietly accept, since the values one of
    them returns are not the values it was given. A gap is the same defect seen
    from the other side: bytes nothing accounts for mean the header does not
    describe this file.

    Entries are ordered by ``(start, end)`` rather than by name, so an empty
    tensor -- zero bytes at the position the next one starts at -- sorts before
    its neighbour instead of appearing to overlap it.

    Args:
        path: the file being read, named in the error.
        parsed: the validated entries, keyed by tensor name.
        data_len: the size of the byte buffer they must cover.

    Raises:
        ValueError: if the ranges overlap, leave a gap, or stop short of the end
            of the buffer.
    """
    offset = 0
    for name, entry in sorted(parsed.items(), key=lambda item: (item[1].start, item[1].end)):
        if entry.start != offset:
            problem = (
                "overlaps the entry before it" if entry.start < offset else "leaves a gap after the entry before it"
            )
            raise ValueError(
                f"{path}: tensor {_short(name)} starts at byte {entry.start} where the buffer is covered "
                f"up to {offset}, so it {problem}."
            )
        offset = entry.end
    if offset != data_len:
        raise ValueError(
            f"{path}: the entries cover {offset} bytes of a {data_len}-byte buffer, "
            f"so {data_len - offset} bytes belong to no tensor."
        )


def check_safetensors(path: str | os.PathLike[str]) -> None:
    """Check that *path* is a readable safetensors file, without reading tensors.

    The header half of :func:`load_safetensors`: the length prefix, the JSON
    header, every entry's dtype, shape and byte range, and that those ranges
    cover the buffer exactly once. A file this accepts is one
    :func:`load_safetensors` can build on the CPU, including an empty tensor's
    shape, which no byte range bounds; another device can add limits of its own.
    It reads only the header, so it stays cheap on a multi-gigabyte checkpoint.

    Written for :func:`kornia.core.download_file_from_url`'s ``validate``
    argument, where it turns a truncated cache entry into a re-download instead
    of a permanent failure. Every truncation the download path can produce is
    visible here: a file cut short after a 2xx leaves the declared header length
    or the entries' ``data_offsets`` pointing past the end.

    Args:
        path: the checkpoint to check.

    Raises:
        ValueError: if the file is not a readable safetensors checkpoint. The
            message names *path*, as :func:`load_safetensors` does.
        OSError: if the file cannot be opened.
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        _read_and_check_header(path, f)


def _read_and_check_header(path: str, f: BinaryIO) -> tuple[dict[str, _TensorEntry], int]:
    """Validate the header of an open safetensors file, leaving nothing mapped.

    Returns:
        The parsed entries, and the offset the byte buffer starts at.
    """
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
    raw = f.read(header_len)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: the header is not valid JSON: {e}") from e
    except ValueError as e:
        # Valid JSON that ``json`` still refuses: an integer with more digits than
        # ``sys.get_int_max_str_digits()`` allows. Its message names no file.
        raise ValueError(f"{path}: the header cannot be decoded: {e}") from e
    except RecursionError as e:
        # The decoder recurses once per nesting level, so about a thousand ``[``
        # -- a kilobyte, far under the header cap -- exhaust the interpreter's
        # recursion limit. A safetensors header nests three levels deep.
        raise ValueError(f"{path}: the header is nested too deeply to decode: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(f"{path}: the header is a JSON {type(header).__name__}, expected an object.")

    entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
    data_len = size - data_start
    parsed = {name: _parse_entry(path, name, entry, data_len) for name, entry in entries.items()}
    _check_the_buffer_is_covered_once(path, parsed, data_len)
    return parsed, data_start


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
            truncated, a header that is not JSON or nests too deeply to decode,
            an entry naming a dtype this reader does not accept, a shape torch
            cannot build, a byte range that does not match the tensor it belongs
            to, or entries that overlap or leave part of the buffer unaccounted
            for. Every message names the file.
        OSError: if the file cannot be opened or mapped.

    Example:
        >>> state_dict = load_safetensors("model.safetensors")  # doctest: +SKIP
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        parsed, data_start = _read_and_check_header(path, f)

        state_dict: dict[str, torch.Tensor] = {}
        # ``ACCESS_COPY`` maps the whole file, so the offsets below are file
        # offsets: the byte buffer starts at ``data_start``.
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as buf:
            for name, (dtype, shape, start, _, numel) in parsed.items():
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
