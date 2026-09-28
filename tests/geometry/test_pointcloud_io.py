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

import os
import struct

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestSaveLoadPointCloud(BaseTester):
    def test_save_pointcloud(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        filename = "pointcloud.ply"
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)

        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        self.assert_close(xyz_save.reshape(-1, 3), xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

    def test_inf_coordinates_save_pointcloud(self, tmp_path):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        xyz_save[0, 0, :] = float("inf")  # all inf → still written, in place
        xyz_save[0, 1, 0] = float("inf")  # partial inf
        xyz_save[1, 0, :-1] = -float("inf")  # partial -inf

        filename = str(tmp_path / "pointcloud_inf.ply")
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)

        # Text mode: the ASCII writer writes platform line endings (``\r\n`` on Windows).
        with open(filename, encoding="utf-8") as f:
            assert "element vertex 80\n" in f.read()
        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        assert xyz_load.shape == (height * width, 3)
        self.assert_close(xyz_load, xyz_save.reshape(-1, 3))

    def test_ascii_writer_round_trips_float64(self, tmp_path):
        # The ASCII writer declares `property double` and converts to float64, so a
        # written value must read back exactly: a fixed significant-digit format
        # loses the digits past the ninth (UTM-scale metres off by millimetres),
        # while the binary writer stores the same values bit-exactly.
        pts = torch.tensor(
            [[0.1234567890123456, -4.5, 1e-7], [5123456.789012, 612345.678901, 123.456789012]],
            dtype=torch.float64,
        )

        filename = str(tmp_path / "pointcloud_f64.ply")
        kornia.geometry.save_pointcloud_ply(filename, pts)

        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        assert xyz_load.dtype == torch.float32, "the loader returns float32"
        # The loader downcasts to float32, so compare through the written text: every
        # float64 value must survive the write as its exact shortest round-trip form.
        with open(filename, encoding="utf-8") as f:
            body = f.read()
        assert body.splitlines()[0] == "ply"
        for value in (0.1234567890123456, 5123456.789012, 612345.678901):
            assert repr(value) in body

    def test_invalid_filename_type(self):
        xyz_save = torch.rand(10, 3)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply(1234, xyz_save)

    def test_invalid_filename_extension(self):
        xyz_save = torch.rand(10, 3)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.txt", xyz_save)

    def test_invalid_pointcloud_type(self):
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.ply", [[1, 2, 3]])

    def test_invalid_pointcloud_shape(self):
        xyz_save = torch.rand(10, 4)
        with pytest.raises(TypeError):
            kornia.geometry.save_pointcloud_ply("pointcloud.ply", xyz_save)

    def test_save_pointcloud_with_nan(self, tmp_path):
        xyz_save = torch.rand(5, 3)
        xyz_save[0, :] = float("nan")  # all nan → still written, in place
        xyz_save[1, 0] = float("nan")
        filename = str(tmp_path / "pointcloud_nan.ply")
        kornia.geometry.save_pointcloud_ply(filename, xyz_save)

        # Text mode: the ASCII writer writes platform line endings (``\r\n`` on Windows).
        with open(filename, encoding="utf-8") as f:
            assert "element vertex 5\n" in f.read()
        xyz_load = kornia.geometry.load_pointcloud_ply(filename)
        assert xyz_load.shape == (5, 3)
        torch.testing.assert_close(xyz_load, xyz_save, equal_nan=True)

    def test_save_pointcloud_binary(self):
        height, width = 10, 8
        xyz_save = torch.rand(height, width, 3)

        filename = "pointcloud_binary.ply"
        kornia.geometry.save_pointcloud_ply_binary(filename, xyz_save)

        xyz_load = kornia.geometry.load_pointcloud_ply_binary(filename)
        self.assert_close(xyz_save.reshape(-1, 3), xyz_load)

        if os.path.exists(filename):
            os.remove(filename)

    def test_save_pointcloud_binary_with_nan_inf(self, tmp_path):
        xyz_save = torch.rand(5, 3)
        xyz_save[0, :] = float("nan")  # all nan → still written, in place
        xyz_save[1, 0] = float("inf")
        xyz_save[3, :] = torch.tensor([float("inf"), -float("inf"), float("nan")])  # all non-finite, mixed
        filename = str(tmp_path / "pointcloud_binary_nan_inf.ply")
        kornia.geometry.save_pointcloud_ply_binary(filename, xyz_save)

        with open(filename, "rb") as f:
            assert b"element vertex 5\n" in f.read()
        xyz_load = kornia.geometry.load_pointcloud_ply_binary(filename)
        assert xyz_load.shape == (5, 3)
        torch.testing.assert_close(xyz_load, xyz_save, equal_nan=True)

    @pytest.mark.parametrize(
        "saver, loader",
        [
            ("save_pointcloud_ply", "load_pointcloud_ply"),
            ("save_pointcloud_ply_binary", "load_pointcloud_ply_binary"),
        ],
    )
    def test_organised_cloud_keeps_invalid_rows_in_place(self, tmp_path, saver, loader):
        # An organised (H, W, 3) cloud, as from `depth_to_3d` with invalid depth: row h * W + w of the
        # file must stay pixel (h, w), so rows whose three coordinates are all non-finite are kept.
        height, width = 3, 4
        xyz_save = torch.arange(height * width * 3, dtype=torch.float32).reshape(height, width, 3) + 0.5
        xyz_save[1, 1, :] = float("nan")
        xyz_save[1, 2, :] = float("inf")
        xyz_save[2, 0, :] = torch.tensor([float("nan"), -float("inf"), float("inf")])
        xyz_save[0, 3, 1] = float("nan")  # a partly finite row as well

        filename = str(tmp_path / f"organised_{saver}.ply")
        getattr(kornia.geometry, saver)(filename, xyz_save)
        xyz_load = getattr(kornia.geometry, loader)(filename)

        assert xyz_load.shape == (height * width, 3)
        torch.testing.assert_close(xyz_load.reshape(height, width, 3), xyz_save, equal_nan=True)
        # The finite rows around the invalid ones did not shift.
        torch.testing.assert_close(xyz_load[4 + 3], xyz_save[1, 3])
        torch.testing.assert_close(xyz_load[2 * 4 + 1], xyz_save[2, 1])


def _ply_header(fmt: str, count: int, properties: str, extra: str = "") -> bytes:
    return (f"ply\nformat {fmt} 1.0\nelement vertex {count}\n{properties}{extra}end_header\n").encode("ascii")


_XYZ_DOUBLE = "property double x\nproperty double y\nproperty double z\n"


class TestLoadPointCloudPlyHeaderParsing(BaseTester):
    """Pins for the header-driven readers (replaces the ``header_size`` line-skipping loaders)."""

    def test_binary_standard_seven_line_header(self, tmp_path):
        # No comment line: the writer emits 8 header lines, a minimal PLY has 7. The old loader
        # skipped 8 lines unconditionally and read the first point as header bytes.
        filename = tmp_path / "minimal.ply"
        filename.write_bytes(_ply_header("binary_little_endian", 2, _XYZ_DOUBLE) + struct.pack("<6d", *range(1, 7)))
        actual = kornia.geometry.load_pointcloud_ply_binary(str(filename))
        self.assert_close(actual, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_binary_stops_at_declared_vertex_count(self, tmp_path):
        # A face element after the vertices must not be read as points.
        filename = tmp_path / "faces.ply"
        payload = struct.pack("<3d", 1.0, 2.0, 3.0) + struct.pack("<B3i", 3, 0, 0, 0)
        header = _ply_header(
            "binary_little_endian", 1, _XYZ_DOUBLE, "element face 1\nproperty list uchar int vertex_indices\n"
        )
        filename.write_bytes(header + payload)
        self.assert_close(kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[1.0, 2.0, 3.0]]))

    def test_binary_extra_vertex_properties_homogeneous(self, tmp_path):
        # x y z nx ny nz, all double: the fast column-select path.
        filename = tmp_path / "normals.ply"
        props = _XYZ_DOUBLE + "property double nx\nproperty double ny\nproperty double nz\n"
        filename.write_bytes(_ply_header("binary_little_endian", 2, props) + struct.pack("<12d", *range(12)))
        self.assert_close(
            kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]])
        )

    def test_binary_mixed_vertex_properties(self, tmp_path):
        # float coordinates followed by uchar colours, with z declared before x: struct path + name lookup.
        filename = tmp_path / "colours.ply"
        props = "property float z\nproperty float x\nproperty float y\n"
        props += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        payload = struct.pack("<3f3B", 3.0, 1.0, 2.0, 255, 0, 7) + struct.pack("<3f3B", 6.0, 4.0, 5.0, 1, 2, 3)
        filename.write_bytes(_ply_header("binary_little_endian", 2, props) + payload)
        self.assert_close(
            kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    def test_binary_big_endian(self, tmp_path):
        filename = tmp_path / "big.ply"
        filename.write_bytes(_ply_header("binary_big_endian", 1, _XYZ_DOUBLE) + struct.pack(">3d", 1.5, -2.0, 3.25))
        self.assert_close(kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[1.5, -2.0, 3.25]]))

    def test_binary_skips_preceding_scalar_element(self, tmp_path):
        filename = tmp_path / "camera_first.ply"
        header = _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(
            b"element vertex", b"element camera 1\nproperty float fx\nproperty float fy\nelement vertex"
        )
        filename.write_bytes(header + struct.pack("<2f", 500.0, 500.0) + struct.pack("<3d", 1.0, 2.0, 3.0))
        self.assert_close(kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[1.0, 2.0, 3.0]]))

    def test_binary_empty(self, tmp_path):
        filename = tmp_path / "empty.ply"
        kornia.geometry.save_pointcloud_ply_binary(str(filename), torch.empty(0, 3))
        actual = kornia.geometry.load_pointcloud_ply_binary(str(filename))
        assert actual.shape == (0, 3)
        assert actual.dtype == torch.float32

    def test_binary_long_header_with_comments(self, tmp_path):
        filename = tmp_path / "comments.ply"
        extra = "".join(f"comment line {i}\n" for i in range(20000)) + "obj_info anything\n"
        filename.write_bytes(_ply_header("binary_little_endian", 1, _XYZ_DOUBLE, extra) + struct.pack("<3d", 1, 2, 3))
        self.assert_close(kornia.geometry.load_pointcloud_ply_binary(str(filename)), torch.tensor([[1.0, 2.0, 3.0]]))

    @pytest.mark.parametrize(
        "header, match",
        [
            (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + _XYZ_DOUBLE.encode(), "no 'end_header'"),
            (_ply_header("binary_little_endian", -1, _XYZ_DOUBLE), "non-negative"),
            (_ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"vertex 1", b"vertex lots"), "Invalid PLY"),
            (_ply_header("binary_little_endian", 1, "property double x\nproperty double y\n"), "x, y and z"),
            (_ply_header("ascii", 1, _XYZ_DOUBLE), "use `load_pointcloud_ply`"),
            (b"not a ply\n" + _ply_header("binary_little_endian", 1, _XYZ_DOUBLE), "first line must be 'ply'"),
            (_ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"format", b"frmt"), "Unknown PLY header"),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"binary_little_endian", b"binary_weird"),
                "Unsupported PLY format line",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"format binary_little_endian 1.0\n", b""),
                "no 'format' line",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"vertex 1", b"vertex"),
                "Malformed PLY element",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(b"vertex 1", b"vertex 1 extra"),
                "Malformed PLY element",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(
                    b"element vertex", b"property double w\nelement vertex"
                ),
                "property before any element",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(
                    b"property double x", b"property quadruple x"
                ),
                "Malformed PLY property",
            ),
            (
                _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(
                    b"property double x", b"property list uchar tags"
                ),
                "Malformed PLY property",
            ),
        ],
    )
    def test_binary_rejects_malformed_header(self, tmp_path, header, match):
        filename = tmp_path / "bad.ply"
        filename.write_bytes(header)  # every case fails before the payload is read
        with pytest.raises(ValueError, match=match):
            kornia.geometry.load_pointcloud_ply_binary(str(filename))

    def test_binary_rejects_truncated_payload(self, tmp_path):
        filename = tmp_path / "short.ply"
        filename.write_bytes(_ply_header("binary_little_endian", 2, _XYZ_DOUBLE) + struct.pack("<3d", 1, 2, 3))
        with pytest.raises(ValueError, match="declares 2 vertices"):
            kornia.geometry.load_pointcloud_ply_binary(str(filename))

    def test_binary_rejects_list_before_vertices(self, tmp_path):
        filename = tmp_path / "face_first.ply"
        header = _ply_header("binary_little_endian", 1, _XYZ_DOUBLE).replace(
            b"element vertex", b"element face 1\nproperty list uchar int vertex_indices\nelement vertex"
        )
        filename.write_bytes(header + bytes(40))
        with pytest.raises(ValueError, match="list property"):
            kornia.geometry.load_pointcloud_ply_binary(str(filename))

    def test_ascii_extra_columns_and_faces(self, tmp_path):
        filename = tmp_path / "ascii_faces.ply"
        props = _XYZ_DOUBLE + "property uchar red\n"
        header = _ply_header("ascii", 2, props, "element face 1\nproperty list uchar int vertex_indices\n")
        filename.write_bytes(header + b"1 2 3 255\n4 5 6 0\n3 0 1 0\n")
        self.assert_close(
            kornia.geometry.load_pointcloud_ply(str(filename)), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    def test_ascii_standard_seven_line_header(self, tmp_path):
        filename = tmp_path / "ascii_minimal.ply"
        filename.write_bytes(_ply_header("ascii", 1, _XYZ_DOUBLE) + b"1 2 3\n")
        self.assert_close(kornia.geometry.load_pointcloud_ply(str(filename)), torch.tensor([[1.0, 2.0, 3.0]]))

    def test_ascii_rejects_binary_file(self, tmp_path):
        filename = tmp_path / "bin.ply"
        kornia.geometry.save_pointcloud_ply_binary(str(filename), torch.ones(1, 3))
        with pytest.raises(ValueError, match="use `load_pointcloud_ply_binary`"):
            kornia.geometry.load_pointcloud_ply(str(filename))

    def test_ascii_rejects_short_vertex_line(self, tmp_path):
        filename = tmp_path / "ascii_short.ply"
        filename.write_bytes(_ply_header("ascii", 1, _XYZ_DOUBLE) + b"1 2\n")
        with pytest.raises(ValueError, match="vertex 0 has 2 values"):
            kornia.geometry.load_pointcloud_ply(str(filename))

    def test_ascii_rejects_truncated_payload(self, tmp_path):
        # A file that simply ends is a truncation, not a malformed line: it must not be reported as one.
        filename = tmp_path / "ascii_trunc.ply"
        filename.write_bytes(_ply_header("ascii", 3, _XYZ_DOUBLE) + b"1 2 3\n")
        with pytest.raises(ValueError, match="declares 3 vertices but the payload ends after 1 of them"):
            kornia.geometry.load_pointcloud_ply(str(filename))

    def test_ascii_rejects_truncated_preceding_element(self, tmp_path):
        filename = tmp_path / "ascii_trunc_camera.ply"
        header = _ply_header("ascii", 1, _XYZ_DOUBLE).replace(
            b"element vertex", b"element camera 2\nproperty float fx\nelement vertex"
        )
        filename.write_bytes(header + b"500\n")
        with pytest.raises(ValueError, match="ends inside element 'camera'"):
            kornia.geometry.load_pointcloud_ply(str(filename))

    def test_ascii_rejects_non_numeric_coordinate(self, tmp_path):
        filename = tmp_path / "ascii_nan_token.ply"
        filename.write_bytes(_ply_header("ascii", 2, _XYZ_DOUBLE) + b"1 2 3\n4 5 oops\n")
        with pytest.raises(ValueError, match="non-numeric coordinate on vertex 1"):
            kornia.geometry.load_pointcloud_ply(str(filename))

    def test_ascii_skips_preceding_scalar_element(self, tmp_path):
        filename = tmp_path / "ascii_camera_first.ply"
        header = _ply_header("ascii", 1, _XYZ_DOUBLE).replace(
            b"element vertex", b"element camera 2\nproperty float fx\nelement vertex"
        )
        filename.write_bytes(header + b"500\n600\n7 8 9\n")
        self.assert_close(kornia.geometry.load_pointcloud_ply(str(filename)), torch.tensor([[7.0, 8.0, 9.0]]))

    def test_ascii_empty(self, tmp_path):
        filename = tmp_path / "ascii_empty.ply"
        kornia.geometry.save_pointcloud_ply(str(filename), torch.empty(0, 3))
        actual = kornia.geometry.load_pointcloud_ply(str(filename))
        assert actual.shape == (0, 3)
        assert actual.dtype == torch.float32

    @pytest.mark.parametrize("loader", ["load_pointcloud_ply", "load_pointcloud_ply_binary"])
    def test_rejects_list_property_in_vertex_element(self, tmp_path, loader):
        # A list spans a variable number of tokens (ASCII) or an unknowable number of bytes (binary),
        # so x, y and z cannot be located after it. Both readers must reject it rather than read the
        # wrong columns: the ASCII reader used to return the tags as coordinates.
        filename = tmp_path / "list_vertex.ply"
        fmt = "ascii" if loader == "load_pointcloud_ply" else "binary_little_endian"
        props = "property list uchar int tags\n" + _XYZ_DOUBLE
        payload = b"2 10 20 1 2 3\n" if fmt == "ascii" else b"\x02" + struct.pack("<2i3d", 10, 20, 1, 2, 3)
        filename.write_bytes(_ply_header(fmt, 1, props) + payload)
        with pytest.raises(ValueError, match="has a list property 'tags'"):
            getattr(kornia.geometry, loader)(str(filename))

    @pytest.mark.parametrize("loader", ["load_pointcloud_ply", "load_pointcloud_ply_binary"])
    def test_rejects_missing_vertex_element(self, tmp_path, loader):
        filename = tmp_path / "no_vertex.ply"
        fmt = "ascii" if loader == "load_pointcloud_ply" else "binary_little_endian"
        filename.write_bytes(
            f"ply\nformat {fmt} 1.0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n".encode(
                "ascii"
            )
        )
        with pytest.raises(ValueError, match="declares no 'vertex' element"):
            getattr(kornia.geometry, loader)(str(filename))

    @pytest.mark.parametrize(
        "loader, saver",
        [("load_pointcloud_ply", "save_pointcloud_ply"), ("load_pointcloud_ply_binary", "save_pointcloud_ply_binary")],
    )
    def test_header_size_is_deprecated_and_ignored(self, tmp_path, loader, saver):
        filename = tmp_path / "compat.ply"
        getattr(kornia.geometry, saver)(str(filename), torch.tensor([[1.0, 2.0, 3.0]]))
        with pytest.warns(DeprecationWarning, match="header_size"):
            actual = getattr(kornia.geometry, loader)(str(filename), header_size=3)
        self.assert_close(actual, torch.tensor([[1.0, 2.0, 3.0]]))
