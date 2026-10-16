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

"""Tests for the pure-torch ``.safetensors`` reader.

Every file here is written **by hand** from the format specification
(https://github.com/huggingface/safetensors#format) rather than by the
``safetensors`` package. That package is a transitive dependency of the dev
environment, so a test that used it to produce the fixtures would pass locally
and prove nothing about the install kornia actually declares -- and a test that
used it to produce the *expectations* would only be comparing two readers.
"""

from __future__ import annotations

import json
import re
import struct
import time
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from kornia.core.safetensors import check_safetensors, load_safetensors

# Names the format gives the dtypes this reader accepts, for the fixtures below.
_NAMES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _raw(tensor: torch.Tensor) -> bytes:
    """Return a tensor's little-endian row-major bytes.

    ``view(torch.uint8)`` reinterprets the storage without touching it, so this
    is the tensor's own bytes rather than a re-encoding of its values -- which is
    what makes the round-trip assertions below exact.
    """
    if tensor.numel() == 0:
        return b""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _build(tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """Serialise *tensors* into safetensors bytes, straight from the spec."""
    header: dict[str, Any] = {}
    blob = b""
    for name, tensor in tensors.items():
        payload = _raw(tensor)
        header[name] = {
            "dtype": _NAMES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(blob), len(blob) + len(payload)],
        }
        blob += payload
    if metadata is not None:
        header["__metadata__"] = metadata
    return _pack(header, blob)


def _pack(header: Any, blob: bytes) -> bytes:
    """Assemble a file from a header and a byte buffer, valid or not.

    The header is an object to serialise, or raw bytes for a header that no
    ``json.dumps`` call would produce.
    """
    encoded = header if isinstance(header, bytes) else json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + blob


def _write(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(payload)
    return path


# ``check_safetensors`` is the header half of ``load_safetensors``: a header one
# of them accepts and the other cannot read is a disagreement, so a header case
# runs through both.
_BOTH_READERS = pytest.mark.parametrize("reader", [check_safetensors, load_safetensors], ids=["check", "load"])


@pytest.fixture
def tensors() -> dict[str, torch.Tensor]:
    """One tensor per interesting case: a float grid, bf16, int64 and an empty one."""
    return {
        "weight": torch.randn(3, 4),
        # bfloat16 has no numpy equivalent, so a reader that goes through numpy
        # cannot serve it at all -- and its 16-bit values are easy to read as
        # float16 by mistake, which would be silently wrong rather than an error.
        "scale": torch.tensor([1.5, -2.0, 0.25], dtype=torch.bfloat16),
        "index": torch.arange(6, dtype=torch.int64).reshape(2, 3),
        # A dimension of 0 stores no bytes but keeps its shape in the header.
        "empty": torch.zeros(0, 5),
    }


class TestLoadSafetensors:
    def test_round_trip(self, tmp_path, tensors) -> None:
        path = _write(tmp_path, _build(tensors, {"format": "pt"}))

        loaded = load_safetensors(path)

        assert list(loaded) == list(tensors), "the header order is the state dict order"
        for name, expected in tensors.items():
            assert loaded[name].dtype == expected.dtype, name
            assert loaded[name].shape == expected.shape, name
            assert torch.equal(loaded[name], expected), name

    def test_metadata_is_not_a_tensor(self, tmp_path, tensors) -> None:
        path = _write(tmp_path, _build(tensors, {"format": "pt"}))

        assert "__metadata__" not in load_safetensors(path)

    def test_reads_a_path_object(self, tmp_path, tensors) -> None:
        path = _write(tmp_path, _build(tensors))

        assert torch.equal(load_safetensors(Path(path))["weight"], tensors["weight"])

    def test_emits_no_warning(self, tmp_path, tensors) -> None:
        """``torch.frombuffer`` warns on a read-only buffer; the mapping must not be one."""
        path = _write(tmp_path, _build(tensors))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_safetensors(path)

        assert not caught, f"load_safetensors warned: {[str(record.message) for record in caught]}"

    def test_tensors_do_not_share_the_file_buffer(self, tmp_path, tensors) -> None:
        """A returned tensor must own its storage, not view a mapping that is closed.

        Writing through a view of the closed mapping is what would crash, and a
        view of a mapping that is *not* closed would keep the file open for as
        long as the model lives.
        """
        path = _write(tmp_path, _build(tensors))

        loaded = load_safetensors(path)
        loaded["weight"] += 1.0
        loaded["index"][0, 0] = 42

        assert loaded["weight"][0, 0] == tensors["weight"][0, 0] + 1.0
        assert loaded["index"][0, 0] == 42
        # The file is untouched: a copy-on-write mapping never reaches the disk,
        # and re-reading it returns what was written.
        assert torch.equal(load_safetensors(path)["index"], tensors["index"])

    @pytest.mark.parametrize("dtype", sorted(_NAMES, key=lambda d: _NAMES[d]), ids=lambda d: _NAMES[d])
    def test_every_supported_dtype_round_trips(self, tmp_path, dtype) -> None:
        expected = torch.tensor([1, 0, 1, 1], dtype=dtype).reshape(2, 2)
        path = _write(tmp_path, _build({"t": expected}))

        loaded = load_safetensors(path)["t"]

        assert loaded.dtype == dtype
        assert torch.equal(loaded, expected)

    def test_device_is_honoured(self, tmp_path, device, tensors) -> None:
        path = _write(tmp_path, _build(tensors))

        loaded = load_safetensors(path, device=device)

        assert loaded["weight"].device.type == torch.device(device).type
        # The empty tensor takes a different branch and must land on the same device.
        assert loaded["empty"].device.type == torch.device(device).type
        assert torch.equal(loaded["weight"].cpu(), tensors["weight"])


class TestRejectsCorruptFiles:
    """Every rejection names the file, so the caller knows what to delete."""

    def test_corrupt_data_offsets(self, tmp_path, tensors) -> None:
        header = {
            "weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 40]},  # 3x4 F32 is 48 bytes
        }
        path = _write(tmp_path, _pack(header, _raw(tensors["weight"])))

        with pytest.raises(ValueError, match="48 bytes, but its data_offsets span 40"):
            load_safetensors(path)

    def test_offsets_outside_the_buffer(self, tmp_path, tensors) -> None:
        header = {"weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [16, 64]}}
        path = _write(tmp_path, _pack(header, _raw(tensors["weight"])))

        with pytest.raises(ValueError, match="not a range inside it"):
            load_safetensors(path)

    def test_reversed_offsets(self, tmp_path, tensors) -> None:
        header = {"weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [48, 0]}}
        path = _write(tmp_path, _pack(header, _raw(tensors["weight"])))

        with pytest.raises(ValueError, match="not a range inside it"):
            load_safetensors(path)

    def test_overlapping_ranges(self, tmp_path) -> None:
        """Two tensors sharing bytes: both ranges fit, and the file is still nonsense."""
        header = {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        }
        path = _write(tmp_path, _pack(header, b"\x00" * 12))

        with pytest.raises(ValueError, match="overlaps the entry before it"):
            load_safetensors(path)

    def test_a_gap_between_ranges(self, tmp_path) -> None:
        """The spec forbids holes: bytes nothing indexes mean the header is not this file's."""
        header = {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [12, 20]},
        }
        path = _write(tmp_path, _pack(header, b"\x00" * 20))

        with pytest.raises(ValueError, match="leaves a gap after the entry before it"):
            load_safetensors(path)

    def test_trailing_bytes_belong_to_no_tensor(self, tmp_path, tensors) -> None:
        path = _write(tmp_path, _build(tensors) + b"\x00" * 16)

        with pytest.raises(ValueError, match="belong to no tensor"):
            load_safetensors(path)

    def test_unknown_dtype(self, tmp_path) -> None:
        header = {"weight": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}}
        path = _write(tmp_path, _pack(header, b"\x00\x00"))

        with pytest.raises(ValueError, match="dtype 'F8_E4M3', which this reader does not support"):
            load_safetensors(path)

    def test_missing_field(self, tmp_path) -> None:
        header = {"weight": {"dtype": "F32", "shape": [1]}}
        path = _write(tmp_path, _pack(header, b"\x00" * 4))

        with pytest.raises(ValueError, match="missing data_offsets"):
            load_safetensors(path)

    def test_entry_is_not_an_object(self, tmp_path) -> None:
        path = _write(tmp_path, _pack({"weight": [1, 2, 3]}, b""))

        with pytest.raises(ValueError, match="header entry 'weight' is list"):
            load_safetensors(path)

    @_BOTH_READERS
    @pytest.mark.parametrize("shape", [[-1], "4", [1.5], [True], None, {"0": 1}, [[1]], ["1"]])
    def test_invalid_shape(self, tmp_path, reader, shape) -> None:
        header = {"weight": {"dtype": "U8", "shape": shape, "data_offsets": [0, 1]}}
        path = _write(tmp_path, _pack(header, b"\x00"))

        with pytest.raises(ValueError, match="expected a list of non-negative integers"):
            reader(path)

    @_BOTH_READERS
    @pytest.mark.parametrize(
        "offsets", [[0], [0, 1, 2], [-1, 1], "0,1", [False, True], None, [0.0, 1.0], [0, "1"], {"0": 0, "1": 1}]
    )
    def test_invalid_offsets(self, tmp_path, reader, offsets) -> None:
        """``[False, True]`` is the offsets twin of the ``[True]`` shape: a bool is an ``int``."""
        header = {"weight": {"dtype": "U8", "shape": [1], "data_offsets": offsets}}
        path = _write(tmp_path, _pack(header, b"\x00"))

        with pytest.raises(ValueError, match="expected two non-negative integers"):
            reader(path)

    @_BOTH_READERS
    @pytest.mark.parametrize(
        "dtype", [["F32"], {"F32": 1}, None, 32, True], ids=["list", "object", "null", "number", "bool"]
    )
    def test_dtype_that_is_not_a_string(self, tmp_path, reader, dtype) -> None:
        """A list or an object is unhashable, so a bare ``in _DTYPES`` raised ``TypeError`` (#5218)."""
        header = {"weight": {"dtype": dtype, "shape": [2], "data_offsets": [0, 8]}}
        path = _write(tmp_path, _pack(header, b"\x00" * 8))

        expected = f"{path}: tensor 'weight' has dtype {dtype!r}, which this reader does not support"
        with pytest.raises(ValueError, match=re.escape(expected)):
            reader(path)

    @_BOTH_READERS
    @pytest.mark.parametrize("where", ["header", "entry"])
    def test_deeply_nested_header(self, tmp_path, reader, where) -> None:
        """200 000 levels is a 400 kB header, far under the cap, and deeper than ``json`` recurses (#5218)."""
        nested = b"[" * 200_000 + b"]" * 200_000
        if where == "entry":
            nested = b'{"weight": {"dtype": ' + nested + b', "shape": [1], "data_offsets": [0, 1]}}'
        path = _write(tmp_path, _pack(nested, b"\x00"))

        with pytest.raises(ValueError, match=re.escape(f"{path}: the header is nested too deeply to decode")):
            reader(path)

    @_BOTH_READERS
    def test_an_integer_too_long_to_decode(self, tmp_path, reader) -> None:
        """Valid JSON that ``json`` still refuses: more digits than ``sys.get_int_max_str_digits()``.

        The ``ValueError`` ``json`` raises for it carries no path, so only the
        path is matched: under a raised digit limit the integer decodes and the
        shape check rejects it instead, with the same prefix.
        """
        header = b'{"weight": {"dtype": "U8", "shape": [' + b"1" * 5000 + b'], "data_offsets": [0, 1]}}'
        path = _write(tmp_path, _pack(header, b"\x00"))

        with pytest.raises(ValueError, match=re.escape(f"{path}: ")):
            reader(path)

    def test_header_is_not_json(self, tmp_path) -> None:
        payload = struct.pack("<Q", 4) + b"nope"
        path = _write(tmp_path, payload)

        with pytest.raises(ValueError, match="the header is not valid JSON"):
            load_safetensors(path)

    def test_header_is_not_an_object(self, tmp_path) -> None:
        path = _write(tmp_path, _pack([1, 2], b""))

        with pytest.raises(ValueError, match="the header is a JSON list"):
            load_safetensors(path)

    def test_truncated_header(self, tmp_path, tensors) -> None:
        payload = _build(tensors)
        path = _write(tmp_path, payload[: 8 + 4])

        with pytest.raises(ValueError, match="but the file holds"):
            load_safetensors(path)

    def test_file_shorter_than_the_length_prefix(self, tmp_path) -> None:
        path = _write(tmp_path, b"\x00\x00\x00")

        with pytest.raises(ValueError, match="too short to be a safetensors file"):
            load_safetensors(path)

    def test_absurd_header_length_is_not_read(self, tmp_path) -> None:
        """A 64-bit length is read before anything is validated; it must be bounded."""
        path = _write(tmp_path, struct.pack("<Q", 2**63) + b"{}")

        with pytest.raises(ValueError, match="more than the"):
            load_safetensors(path)

    def test_the_message_names_the_file(self, tmp_path) -> None:
        header = {"weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 40]}}
        path = _write(tmp_path, _pack(header, b"\x00" * 48))

        with pytest.raises(ValueError, match=re.escape(str(path))):
            load_safetensors(path)


class TestShapesTorchCannotBuild:
    """``check_safetensors`` accepts exactly the shapes ``load_safetensors`` can build on the CPU (#5218).

    An empty tensor stores no bytes, so the byte-size check says nothing about
    its other dimensions. The format allows empty tensors because they are
    valid in the tensor libraries, and a torch tensor's sizes and contiguous
    strides are 64-bit signed integers: a shape outside that is not a torch
    tensor. Both readers reject it in the header check, with a message naming
    the file, instead of the load failing inside ``torch.empty``.
    """

    @_BOTH_READERS
    @pytest.mark.parametrize("shape", [[2**63, 0], [0, 2**63], [2**64 - 1, 0]])
    def test_a_dimension_beyond_int64(self, tmp_path, reader, shape) -> None:
        header = {"weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}}
        path = _write(tmp_path, _pack(header, b""))

        with pytest.raises(ValueError, match=re.escape(f"{path}: tensor 'weight' has shape {shape}, whose dimensions")):
            reader(path)

    @_BOTH_READERS
    @pytest.mark.parametrize("shape", [[0, 2**62, 2**62], [0, 3, 2**62]])
    def test_an_empty_shape_whose_strides_overflow(self, tmp_path, reader, shape) -> None:
        """Every dimension fits in 64 bits; the contiguous strides torch derives from them do not."""
        header = {"weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}}
        path = _write(tmp_path, _pack(header, b""))

        with pytest.raises(ValueError, match=re.escape(f"{path}: tensor 'weight' has shape {shape}, which torch")):
            reader(path)

    @_BOTH_READERS
    @pytest.mark.parametrize("shape", [[2**62, 2**62, 0], [2**62, 4, 0]])
    def test_an_empty_shape_whose_storage_size_may_overflow(self, tmp_path, reader, shape) -> None:
        """The readers accept these exactly when ``torch.empty`` builds them on the CPU.

        Whether it does depends on how c10 was compiled. Its storage-size check
        (``safe_multiplies_u64`` in ``c10/util/safe_numerics.h``) multiplies the
        sizes in order in unsigned 64 bits where the compiler provides a checked
        multiply, which rejects both shapes on the macOS build this was written
        on; the header's MSVC branch counts any zero size as no overflow. The
        test pins agreement with the running build rather than either answer.
        """
        try:
            torch.empty(shape, dtype=torch.float32)
        except RuntimeError:
            torch_builds_it = False
        else:
            torch_builds_it = True
        header = {"weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}}
        path = _write(tmp_path, _pack(header, b""))

        if torch_builds_it:
            reader(path)
        else:
            with pytest.raises(ValueError, match=re.escape(f"{path}: tensor 'weight' has shape {shape}, which torch")):
                reader(path)

    @pytest.mark.parametrize("shape", [[2**63 - 1, 0], [0, 2**63 - 1], [2**62, 3, 0]])
    def test_an_empty_tensor_keeps_a_huge_nominal_shape(self, tmp_path, device, shape) -> None:
        """The other side of the boundary: what torch can build, both readers accept.

        ``[2**62, 3, 0]`` builds under either branch of c10's storage-size check,
        while ``[2**62, 4, 0]`` depends on the branch (see the test above). That
        is why the check asks torch rather than restating its arithmetic.
        """
        header = {"weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}}
        path = _write(tmp_path, _pack(header, b""))

        check_safetensors(path)
        loaded = load_safetensors(path, device=device)["weight"]

        assert list(loaded.shape) == shape
        assert loaded.device.type == torch.device(device).type

    @_BOTH_READERS
    @pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
    def test_many_large_dimensions_are_rejected_in_linear_time(self, tmp_path, reader, empty) -> None:
        """100 000 dimensions of ``2**62``, a 2 MB header, are rejected without multiplying them all out.

        Multiplying all of them out is quadratic in their count -- close to a
        minute at this size -- and reaches more digits than ``str`` converts.
        The product has to stop once it passes what the buffer holds, which the
        message says, or, for the empty shape whose ``0`` comes last, once it
        passes what a 64-bit stride can hold. The time bound is loose: the
        rejection takes a few hundredths of a second, the quadratic product
        about fifty.
        """
        shape = [2**62] * 100_000 + ([0] if empty else [])
        header = {"weight": {"dtype": "U8", "shape": shape, "data_offsets": [0, 0 if empty else 1]}}
        path = _write(tmp_path, _pack(header, b"" if empty else b"\x00"))
        problem = "which torch cannot build" if empty else "which is more than the 1-byte buffer holds, but its"

        start = time.perf_counter()
        with pytest.raises(ValueError, match=re.escape(f"{path}: tensor 'weight' ")) as caught:
            reader(path)
        elapsed = time.perf_counter() - start

        assert problem in str(caught.value)
        assert len(str(caught.value)) < 2000
        assert elapsed < 10.0, f"rejecting the header took {elapsed:.1f} s"


class TestMessagesStayShort:
    """The file decides how large a header value is; the message quoting it stays a few lines."""

    @_BOTH_READERS
    @pytest.mark.parametrize("field", ["dtype", "shape", "data_offsets", "name"])
    def test_a_huge_value_is_shortened(self, tmp_path, reader, field) -> None:
        """A 10M-element ``dtype`` list is a 20 MB header, under the cap; quoted whole, it was a 30 MB message."""
        entry = {"dtype": b'"U8"', "shape": b"[1]", "data_offsets": b"[0, 1]"}
        name = b"weight"
        if field == "dtype":
            entry["dtype"] = b"[" + b"0," * (10**7 - 1) + b"0]"
        elif field == "shape":
            entry["shape"] = b"[" + b"-1," * (10**6 - 1) + b"-1]"
        elif field == "data_offsets":
            entry["data_offsets"] = b"[" + b"0," * (10**6 - 1) + b"0]"
        else:
            name = b"w" * 10**6
            entry["dtype"] = b'"XYZ"'
        fields = b", ".join(b'"' + key.encode() + b'": ' + value for key, value in entry.items())
        path = _write(tmp_path, _pack(b'{"' + name + b'": {' + fields + b"}}", b"\x00"))

        with pytest.raises(ValueError, match=re.escape(f"{path}: tensor ")) as caught:
            reader(path)

        assert len(str(caught.value)) < 2000, f"a {len(str(caught.value))}-character message"

    def test_a_size_mismatch_states_the_exact_byte_count(self, tmp_path) -> None:
        """The product is exact when it completes, even if its last factor takes it past the buffer."""
        header = {"weight": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}
        path = _write(tmp_path, _pack(header, b"\x00" * 8))

        with pytest.raises(ValueError, match=re.escape("is F32[3], which is 12 bytes, but its data_offsets span 8.")):
            load_safetensors(path)

    def test_a_shape_larger_than_the_buffer_says_so(self, tmp_path) -> None:
        """The product stops once it passes the buffer, so the count is a bound, not a number."""
        header = {"weight": {"dtype": "F32", "shape": [2**40, 2**40], "data_offsets": [0, 8]}}
        path = _write(tmp_path, _pack(header, b"\x00" * 8))

        expected = f"is F32{[2**40, 2**40]}, which is more than the 8-byte buffer holds, but its data_offsets span 8."
        with pytest.raises(ValueError, match=re.escape(expected)):
            load_safetensors(path)
