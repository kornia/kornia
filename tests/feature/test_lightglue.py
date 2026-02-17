import torch
from kornia.feature import LightGlue


def test_lightglue_empty_after_pruning():
    model = LightGlue(features="superpoint", width_confidence=0.99)
    model.eval()

    data = {
        "image0": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),

        },
        "image1": {
            "keypoints": torch.empty(1, 0, 2),
            "descriptors": torch.empty(1, 0, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
    }

    with torch.no_grad():
        out = model(data)

    assert out["matches0"].shape == (1, 0)
    assert out["matches1"].shape == (1, 0)
    assert out["matching_scores0"].shape == (1, 0)
    assert out["matching_scores1"].shape == (1, 0)