from __future__ import annotations

import pytest
import torch


@pytest.fixture
def data_loftr():
    url = 'https://github.com/kornia/data_test/blob/main/loftr_outdoor_and_homography_data.pt?raw=true'
    return torch.hub.load_state_dict_from_url(url)
