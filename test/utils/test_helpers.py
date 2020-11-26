import pytest

import torch

from kornia.utils import _extract_device_dtype


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
