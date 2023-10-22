import pytest
import torch

from kornia.feature import OnnxLightGlue


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OnnxLightGlue requires CUDA")
class TestOnnxLightGlue:
    @pytest.mark.slow
    @pytest.mark.parametrize("weights", OnnxLightGlue.MODEL_URLS.keys())
    def test_pretrained_weights(self, weights):
        model = OnnxLightGlue(weights)
        assert model is not None

    @pytest.mark.slow
    def test_forward(self):
        model = OnnxLightGlue()

        device = torch.device("cuda")
        kpts = torch.zeros(1, 5, 2, device=device)
        desc = torch.zeros(1, 5, 128, device=device)
        image = torch.zeros(1, 3, 10, 10)
        outputs = model(
            {
                "image0": {"keypoints": kpts, "descriptors": desc, "image": image},
                "image1": {"keypoints": kpts, "descriptors": desc, "image": image},
            }
        )

        assert "matches" in outputs
        assert "scores" in outputs

    @pytest.mark.slow
    def test_exception(self):
        with pytest.raises(RuntimeError):
            OnnxLightGlue(device="invalid device")

        model = OnnxLightGlue()

        device = torch.device("cuda")
        kpts = torch.zeros(1, 5, 2, device=device)
        desc = torch.zeros(1, 5, 128, device=device)
        image = torch.zeros(1, 3, 10, 10)

        # Missing input
        with pytest.raises(Exception):
            model({"image0": {"keypoints": kpts, "descriptors": desc, "image": image}})

        # Wrong dtype
        with pytest.raises(Exception):
            model(
                {
                    "image0": {
                        "keypoints": torch.zeros(1, 5, 2, dtype=torch.int32, device=device),
                        "descriptors": desc,
                        "image": image,
                    },
                    "image1": {"keypoints": kpts, "descriptors": desc, "image": image},
                }
            )

        # Wrong device
        with pytest.raises(Exception):
            model(
                {
                    "image0": {"keypoints": torch.zeros(1, 5, 2, device="cpu"), "descriptors": desc, "image": image},
                    "image1": {"keypoints": kpts, "descriptors": desc, "image": image},
                }
            )

        # Wrong shapes
        with pytest.raises(Exception):
            model(
                {
                    "image0": {"keypoints": torch.zeros(1, 4, 2, device=device), "descriptors": desc, "image": image},
                    "image1": {"keypoints": kpts, "descriptors": desc, "image": image},
                }
            )
        with pytest.raises(Exception):
            model(
                {
                    "image0": {"keypoints": kpts, "descriptors": torch.zeros(1, 5, 127, device=device), "image": image},
                    "image1": {"keypoints": kpts, "descriptors": desc, "image": image},
                }
            )
