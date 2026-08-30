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

import array
import os
import struct
import sys
import warnings
from typing import BinaryIO, Dict, List, NamedTuple, Optional, Tuple

import torch


def save_pointcloud_ply(filename: str, pointcloud: torch.Tensor) -> None:
    r"""Save to disk a pointcloud in PLY format.

    Args:
        filename: the path to save the pointcloud.
        pointcloud: tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.
    """
    if not (isinstance(filename, str) and filename.lower().endswith(".ply")):
        raise TypeError(f"Input filename must be a string with the .ply extension. Got {filename!r}")

    if not torch.is_tensor(pointcloud):
        raise TypeError(f"Input pointcloud type is not a torch.Tensor. Got {type(pointcloud)}")

    if pointcloud.ndim < 2 or pointcloud.shape[-1] != 3:
        raise TypeError(f"Input pointcloud must have shape (..., 3). Got {tuple(pointcloud.shape)}")

    # Flatten points
    xyz = pointcloud.reshape(-1, 3)

    valid_mask = torch.isfinite(xyz).any(dim=1)
    valid_points = xyz[valid_mask]
    valid_count = valid_points.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        # Write PLY header
        f.writelines(
            [
                "ply\n",
                "format ascii 1.0\n",
                "comment arraiy generated\n",
                f"element vertex {valid_count}\n",
                "property double x\n",
                "property double y\n",
                "property double z\n",
                "end_header\n",
            ]
        )

        if valid_count > 0:
            # Move to CPU, convert to float64 for matching 'double' in header
            arr = valid_points.detach().cpu().to(torch.float64)
            # Write each row as space-separated floats
            for x, y, z in arr.tolist():
                f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def save_pointcloud_ply_binary(filename: str, pointcloud: torch.Tensor) -> None:
    r"""Save to disk a pointcloud in binary PLY format.

    Args:
        filename: the path to save the pointcloud.
        pointcloud: tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.
    """
    if not (isinstance(filename, str) and filename.lower().endswith(".ply")):
        raise TypeError(f"Input filename must be a string with the .ply extension. Got {filename!r}")

    if not torch.is_tensor(pointcloud):
        raise TypeError(f"Input pointcloud type is not a torch.Tensor. Got {type(pointcloud)}")

    if pointcloud.ndim < 2 or pointcloud.shape[-1] != 3:
        raise TypeError(f"Input pointcloud must have shape (..., 3). Got {tuple(pointcloud.shape)}")

    # Flatten points
    xyz = pointcloud.reshape(-1, 3)

    valid_mask = torch.isfinite(xyz).any(dim=1)
    valid_points = xyz[valid_mask]
    valid_count = valid_points.shape[0]

    with open(filename, "wb") as f:
        # Write PLY header : Binary version
        header = [
            "ply\n",
            "format binary_little_endian 1.0\n",
            "comment kornia generated\n",
            f"element vertex {valid_count}\n",
            "property double x\n",
            "property double y\n",
            "property double z\n",
            "end_header\n",
        ]
        f.writelines(s.encode("utf-8") for s in header)

        if valid_count > 0:
            # Move to CPU, convert to float64 for matching 'double' in header
            arr = valid_points.detach().cpu().to(torch.float64).reshape(-1)

            # Convert to array.array for efficient byte-level handling
            data_array = array.array("d", arr.tolist())

            # Ensure little-endian
            if sys.byteorder == "big":
                data_array.byteswap()

            # Write binary data in a single operation for I/O efficiency
            f.write(data_array.tobytes())


class _PlyElement(NamedTuple):
    name: str
    count: int
    # (property name, scalar type); ``None`` as the type marks a ``property list``
    properties: List[Tuple[str, Optional[str]]]


class _PlyHeader(NamedTuple):
    format: str
    elements: List[_PlyElement]


# PLY scalar types -> (struct code, torch dtype). Both the classic and the sized spellings are legal.
_PLY_SCALAR_TYPES: Dict[str, Tuple[str, Optional[torch.dtype]]] = {
    "char": ("b", torch.int8),
    "int8": ("b", torch.int8),
    "uchar": ("B", torch.uint8),
    "uint8": ("B", torch.uint8),
    "short": ("h", torch.int16),
    "int16": ("h", torch.int16),
    "ushort": ("H", None),
    "uint16": ("H", None),
    "int": ("i", torch.int32),
    "int32": ("i", torch.int32),
    "uint": ("I", None),
    "uint32": ("I", None),
    "float": ("f", torch.float32),
    "float32": ("f", torch.float32),
    "double": ("d", torch.float64),
    "float64": ("d", torch.float64),
}
_PLY_FORMATS = ("ascii", "binary_little_endian", "binary_big_endian")


def _read_ply_header(f: BinaryIO, filename: str) -> _PlyHeader:
    """Parse a PLY header up to and including ``end_header``.

    Leaves the file positioned at the first byte of the element data. The whole header is ASCII by
    definition of the format, whatever the payload encoding.
    """
    if f.readline().strip() != b"ply":
        raise ValueError(f"{filename!r} is not a PLY file: the first line must be 'ply'.")

    fmt: Optional[str] = None
    elements: List[_PlyElement] = []
    while True:
        raw = f.readline()
        if not raw:
            raise ValueError(f"PLY header in {filename!r} has no 'end_header' line.")
        tokens = raw.decode("ascii", errors="replace").split()
        if not tokens:
            continue
        keyword = tokens[0]
        if keyword == "end_header":
            break
        if keyword in ("comment", "obj_info"):
            continue
        if keyword == "format":
            if len(tokens) < 2 or tokens[1] not in _PLY_FORMATS:
                raise ValueError(f"Unsupported PLY format line {raw!r} in {filename!r}.")
            fmt = tokens[1]
        elif keyword == "element":
            if len(tokens) != 3:
                raise ValueError(f"Malformed PLY element line {raw!r} in {filename!r}.")
            try:
                count = int(tokens[2])
            except ValueError as exc:
                raise ValueError(f"Invalid PLY element count {tokens[2]!r} in {filename!r}.") from exc
            if count < 0:
                raise ValueError(f"PLY element count must be non-negative in {filename!r}. Got {count}.")
            elements.append(_PlyElement(tokens[1], count, []))
        elif keyword == "property":
            if not elements:
                raise ValueError(f"PLY property before any element in {filename!r}.")
            if len(tokens) == 3 and tokens[1] in _PLY_SCALAR_TYPES:
                elements[-1].properties.append((tokens[2], tokens[1]))
            elif (
                len(tokens) == 5
                and tokens[1] == "list"
                and tokens[2] in _PLY_SCALAR_TYPES
                and tokens[3] in _PLY_SCALAR_TYPES
            ):
                elements[-1].properties.append((tokens[4], None))
            else:
                raise ValueError(f"Malformed PLY property line {raw!r} in {filename!r}.")
        else:
            raise ValueError(f"Unknown PLY header keyword {keyword!r} in {filename!r}.")

    if fmt is None:
        raise ValueError(f"PLY header in {filename!r} has no 'format' line.")
    return _PlyHeader(fmt, elements)


def _ply_vertex_layout(header: _PlyHeader, filename: str) -> Tuple[int, _PlyElement, Tuple[int, int, int]]:
    """Locate the ``vertex`` element and the column indices of ``x``, ``y`` and ``z``."""
    for index, element in enumerate(header.elements):
        if element.name == "vertex":
            names = [name for name, _ in element.properties]
            try:
                xyz = (names.index("x"), names.index("y"), names.index("z"))
            except ValueError as exc:
                raise ValueError(
                    f"PLY vertex element in {filename!r} must declare x, y and z properties. Got {names}."
                ) from exc
            return index, element, xyz
    raise ValueError(f"PLY file {filename!r} declares no 'vertex' element.")


def _warn_header_size(header_size: Optional[int]) -> None:
    if header_size is not None:
        warnings.warn(
            "`header_size` is ignored since kornia 0.8.3: the PLY header is parsed up to `end_header`. "
            "The argument will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )


def _check_ply_filename(filename: str) -> None:
    if not (isinstance(filename, str) and filename.lower().endswith(".ply")):
        raise TypeError(f"Input filename must be a string with the .ply extension. Got {filename!r}")
    if not os.path.isfile(filename):
        raise ValueError("Input filename is not an existing file.")


def load_pointcloud_ply(filename: str, header_size: Optional[int] = None) -> torch.Tensor:
    r"""Load from disk a pointcloud in ASCII PLY format.

    The header is parsed up to ``end_header``; the number of points comes from the
    ``element vertex N`` declaration and the coordinates from the ``x``, ``y`` and ``z``
    properties, whatever other properties (normals, colours) or elements (faces) the file carries.

    Args:
        filename: the path to the pointcloud.
        header_size: deprecated and ignored; the header is parsed instead of skipped.

    Return:
        tensor containing the loaded points with shape :math:`(N, 3)` in ``float32``, where
        :math:`N` is the declared vertex count.
    """
    _check_ply_filename(filename)
    _warn_header_size(header_size)

    with open(filename, "rb") as f:
        header = _read_ply_header(f, filename)
        if header.format != "ascii":
            raise ValueError(
                f"{filename!r} is a {header.format} PLY file; use `load_pointcloud_ply_binary` to read it."
            )
        vertex_index, vertex, xyz = _ply_vertex_layout(header, filename)

        # In ASCII PLY every element instance is one line, list properties included.
        for element in header.elements[:vertex_index]:
            for _ in range(element.count):
                if not f.readline():
                    raise ValueError(f"PLY file {filename!r} ends inside element {element.name!r}.")

        num_columns = len(vertex.properties)
        points: List[float] = []
        for _ in range(vertex.count):
            tokens = f.readline().split()
            if len(tokens) < num_columns:
                raise ValueError(
                    f"PLY file {filename!r} declares {vertex.count} vertices with {num_columns} properties, "
                    f"but a vertex line has {len(tokens)} values."
                )
            points.extend(float(tokens[i]) for i in xyz)

    return torch.tensor(points, dtype=torch.float32).reshape(-1, 3)


def load_pointcloud_ply_binary(filename: str, header_size: Optional[int] = None) -> torch.Tensor:
    r"""Load from disk a pointcloud in binary PLY format.

    The header is parsed up to ``end_header``; the number of points comes from the
    ``element vertex N`` declaration and the coordinates from the ``x``, ``y`` and ``z``
    properties, whatever their scalar type. Extra vertex properties (normals, colours) are skipped,
    and so are elements that follow the vertices (faces). Both little- and big-endian payloads are
    read. Elements that *precede* the vertices are skipped only when they contain no list property,
    since the byte length of a list is unknown without parsing it.

    Args:
        filename: the path to the pointcloud.
        header_size: deprecated and ignored; the header is parsed instead of skipped.

    Return:
        tensor containing the loaded points with shape :math:`(N, 3)` in ``float32``, where
        :math:`N` is the declared vertex count.
    """
    _check_ply_filename(filename)
    _warn_header_size(header_size)

    with open(filename, "rb") as f:
        header = _read_ply_header(f, filename)
        if header.format == "ascii":
            raise ValueError(f"{filename!r} is an ASCII PLY file; use `load_pointcloud_ply` to read it.")
        endian = "<" if header.format == "binary_little_endian" else ">"
        vertex_index, vertex, xyz = _ply_vertex_layout(header, filename)

        def element_struct(element: _PlyElement) -> str:
            codes = []
            for name, scalar_type in element.properties:
                if scalar_type is None:
                    raise ValueError(
                        f"PLY element {element.name!r} in {filename!r} has a list property {name!r}; "
                        "list properties are only supported after the vertex element."
                    )
                codes.append(_PLY_SCALAR_TYPES[scalar_type][0])
            return endian + "".join(codes)

        for element in header.elements[:vertex_index]:
            if element.count > 0:
                f.seek(element.count * struct.calcsize(element_struct(element)), os.SEEK_CUR)

        vertex_format = element_struct(vertex)
        stride = struct.calcsize(vertex_format)
        expected = vertex.count * stride
        data = f.read(expected)
        if len(data) != expected:
            raise ValueError(
                f"PLY file {filename!r} declares {vertex.count} vertices ({expected} bytes) "
                f"but only {len(data)} bytes follow the header."
            )

    if vertex.count == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    scalar_types = {scalar_type for _, scalar_type in vertex.properties}
    if len(scalar_types) == 1:
        code, dtype = _PLY_SCALAR_TYPES[next(iter(scalar_types))]
        if dtype is not None:
            # Homogeneous layout: one bulk copy, then column-select x, y, z.
            values = array.array(code, data)
            if (endian == "<") != (sys.byteorder == "little"):
                values.byteswap()
            table = torch.frombuffer(values, dtype=dtype).reshape(vertex.count, len(vertex.properties))
            return table[:, list(xyz)].to(torch.float32).clone()

    # Mixed layout (or an unsigned type torch cannot hold): unpack row by row.
    rows = struct.iter_unpack(vertex_format, data)
    return torch.tensor([[row[i] for i in xyz] for row in rows], dtype=torch.float32)
