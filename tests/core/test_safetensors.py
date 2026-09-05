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
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from kornia.core.safetensors import load_safetensors

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
    """Assemble a file from a header object and a byte buffer, valid or not."""
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + blob


def _write(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(payload)
    return path


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

    @pytest.mark.parametrize("shape", [[-1], "4", [1.5], [True]])
    def test_invalid_shape(self, tmp_path, shape) -> None:
        header = {"weight": {"dtype": "U8", "shape": shape, "data_offsets": [0, 1]}}
        path = _write(tmp_path, _pack(header, b"\x00"))

        with pytest.raises(ValueError, match="expected a list of non-negative integers"):
            load_safetensors(path)

    @pytest.mark.parametrize("offsets", [[0], [0, 1, 2], [-1, 1], "0,1"])
    def test_invalid_offsets(self, tmp_path, offsets) -> None:
        header = {"weight": {"dtype": "U8", "shape": [1], "data_offsets": offsets}}
        path = _write(tmp_path, _pack(header, b"\x00"))

        with pytest.raises(ValueError, match="expected two non-negative integers"):
            load_safetensors(path)

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
