import pytest

import torch

from kornia.utils import _extract_device_dtype, _parse_align_corners


@pytest.mark.parametrize("tensor_list,out_device,out_dtype,will_throw_error", [
    ([], torch.device('cpu'), torch.get_default_dtype(), False),
    ([None, None], torch.device('cpu'), torch.get_default_dtype(), False),
    ([torch.tensor(0, device='cpu', dtype=torch.float16), None], torch.device('cpu'), torch.float16, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float32), None], torch.device('cpu'), torch.float32, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float64), None], torch.device('cpu'), torch.float64, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float16)] * 2, torch.device('cpu'), torch.float16, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float32)] * 2, torch.device('cpu'), torch.float32, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float64)] * 2, torch.device('cpu'), torch.float64, False),
    ([torch.tensor(0, device='cpu', dtype=torch.float16),
        torch.tensor(0, device='cpu', dtype=torch.float64)], None, None, True),
    ([torch.tensor(0, device='cpu', dtype=torch.float32),
        torch.tensor(0, device='cpu', dtype=torch.float64)], None, None, True),
    ([torch.tensor(0, device='cpu', dtype=torch.float16),
        torch.tensor(0, device='cpu', dtype=torch.float32)], None, None, True),
])
def test_extract_device_dtype(tensor_list, out_device, out_dtype, will_throw_error):
    # Add GPU tests when GPU testing avaliable
    if torch.cuda.is_available():
        import warnings
        warnings.warn("Add GPU tests.")

    if will_throw_error:
        with pytest.raises(ValueError):
            _extract_device_dtype(tensor_list)
    else:
        device, dtype = _extract_device_dtype(tensor_list)
        assert device == out_device
        assert dtype == out_dtype


interpolations = pytest.mark.parametrize(
    "interpolation",
    ("nearest", "linear", "bilinear", "bicubic", "trilinear", "area"),
    ids=lambda argvalue: f"interpolation={argvalue}"
)


@interpolations
def test_parse_align_corners_default(interpolation):
    align_corners = _parse_align_corners(None, interpolation)
    if interpolation in ("linear", "bilinear", "bicubic", "trilinear"):
        assert align_corners is False
    else:
        assert align_corners is None


@interpolations
@pytest.mark.parametrize("align_corners", (True, False), ids=lambda align_corners: f"align_corners={align_corners}")
def test_parse_align_corners_non_default(align_corners, interpolation):
    assert _parse_align_corners(align_corners, interpolation) is align_corners
