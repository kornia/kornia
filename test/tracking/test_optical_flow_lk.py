import torch

from kornia.tracking.optical_flow_lk import optical_flow_lk


class TestOpticalFlowLK:
    def test_smoke(self):
        img1 = torch.rand(3, 32, 32)
        img2 = torch.rand(3, 32, 32)
        pts1 = torch.rand(10, 2)
        pts2 = optical_flow_lk(img1, img2, pts1)
        assert pts2.shape == (10, 2)

    def test_synthetic_identity(self, device, dtype):
        # Create a synthetic image
        img = torch.zeros(1, 10, 10, device=device, dtype=dtype)
        img[..., 2:8, 2:8] = 1.0

        # Create a synthetic set of points
        pts = torch.tensor(
            [
                [2.0, 2.0],
                [7.0, 2.0],
                [7.0, 7.0],
                [2.0, 7.0],
            ],
            device=device,
            dtype=dtype,
        )

        # Apply LK
        pts2 = optical_flow_lk(img, img, pts)

        # Check the output
        assert torch.allclose(pts2, pts)

    def test_synthetic_translation(self, device, dtype):
        # Create a synthetic image
        # img1 = torch.zeros(1, 10, 10, device=device, dtype=dtype)
        img1 = torch.zeros(1, 10, 10, device=device, dtype=dtype)
        img1[..., 2:8, 2:8] = 1.0
        img1[..., :2, :2] = 1.0
        # img2 = torch.zeros(1, 10, 10, device=device, dtype=dtype)
        img2 = torch.zeros(1, 10, 10, device=device, dtype=dtype)
        img2[..., 3:9, 3:9] = 1.0
        img2[..., :3, :3] = 1.0

        # Create a synthetic set of points
        pts = torch.tensor(
            [
                [2.0, 2.0],
                # [7.0, 2.0],
                # [7.0, 7.0],
                # [2.0, 7.0],
            ],
            device=device,
            dtype=dtype,
        )
        # pts = pts.repeat(100, 1)

        import time

        # optical_flow_lk_compiled = torch.compile(optical_flow_lk)
        # class OpticalFlowLK(torch.nn.Module):
        #    def __init__(self):
        #        super().__init__()
        #        self.optical_flow_lk = optical_flow_lk

        #    def forward(self, img1, img2, pts):
        #        return self.optical_flow_lk(img1, img2, pts)

        # optical_flow_lk_module = OpticalFlowLK()

        # onnx
        # torch.onnx.export(
        #     optical_flow_lk_module,
        #     (img1, img2, pts),
        #     "optical_flow_lk.onnx",
        #     verbose=True
        # )

        # Apply LK
        times = []
        for i in range(1):
            t0 = time.time()
            # pts2 = optical_flow_lk_compiled(img1, img2, pts)
            pts2 = optical_flow_lk(img1, img2, pts, max_level=2)
            t1 = time.time()
            times.append((t1 - t0) * 1000)

        time_mean = sum(times) / len(times)
        print(f"Time mean: {time_mean} ms")

        print(pts2)

        # Check the output
        # assert torch.allclose(pts2, pts + 1.0)
