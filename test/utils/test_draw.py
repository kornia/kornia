import pytest
import torch

from kornia.utils import draw_rectangle


class TestDrawRectangle:
    @pytest.mark.parametrize('batch', (4, 17))
    @pytest.mark.parametrize('color', (torch.Tensor([1.0]), torch.Tensor([0.5])))
    def test_smoke(self, device, batch, color):
        black_image = torch.zeros(batch, 1, 3, 3, device=device)  # 1 channel 3x3 black_image
        points = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device).expand(batch, 1, 4)  # single pixel rectangle

        draw_rectangle(black_image, points, color=color)

        target = torch.zeros(batch, 1, 3, 3, device=device)
        target[:, :, 1, 1] = color

        assert torch.all(black_image == target)

    @pytest.mark.parametrize('batch', (8, 11))
    @pytest.mark.parametrize('fill', (True, False))
    @pytest.mark.parametrize('height', (12, 106, 298))
    @pytest.mark.parametrize('width', (7, 123, 537))
    def test_fill_and_edges(self, device, batch, fill, height, width):
        black_image = torch.zeros(batch, 3, height, width, device=device)
        # we should pass height - 1 and width - 1 but rectangle should clip correctly
        points = torch.tensor([0, 0, width, height]).to(device).expand(batch, 1, 4)

        image_w_rectangle = draw_rectangle(black_image, points, color=torch.tensor([1.0]), fill=fill)

        assert image_w_rectangle is black_image
        if fill:
            assert image_w_rectangle.sum() == batch * 3 * height * width
        else:
            # corners are double counted
            assert image_w_rectangle.sum() == batch * 3 * (2 * height + 2 * width - 4)

    @pytest.mark.parametrize('batch', (4, 6))
    @pytest.mark.parametrize('N', (5, 12))
    @pytest.mark.parametrize('fill', (True, False))
    def test_n_rectangles(self, device, batch, N, fill):
        points_list = []
        h, w = 20, 20
        for b in range(batch):
            points_list.append([])
            for n in range(N):
                points_list[b].append([])
                points_list[b][n].append(int(torch.randint(0, w - 1, (1,))))
                points_list[b][n].append(int(torch.randint(0, h - 1, (1,))))
                points_list[b][n].append(int(torch.randint(points_list[b][n][-2] + 1, w, (1,))))
                points_list[b][n].append(int(torch.randint(points_list[b][n][-2] + 1, h, (1,))))

        points = torch.tensor(points_list).to(device)

        random_background = torch.rand(batch, 3, h, w, device=device)
        random_w_rectangle = random_background.clone()

        draw_rectangle(random_w_rectangle, points, color=torch.tensor([1.0, 1.0, 1.0]), fill=fill)

        for b in range(batch):
            for n in range(N):
                if fill:
                    assert (
                        random_w_rectangle[
                            b,
                            :,
                            points_list[b][n][1] : points_list[b][n][3] + 1,
                            points_list[b][n][0] : points_list[b][n][2] + 1,
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1)
                        * (points_list[b][n][2] - points_list[b][n][0] + 1)
                        * 3
                    )
                else:
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1] : points_list[b][n][3] + 1, points_list[b][n][0]
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1] : points_list[b][n][3] + 1, points_list[b][n][2]
                        ].sum()
                        == (points_list[b][n][3] - points_list[b][n][1] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1], points_list[b][n][0] : points_list[b][n][2] + 1
                        ].sum()
                        == (points_list[b][n][2] - points_list[b][n][0] + 1) * 3
                    )
                    assert (
                        random_w_rectangle[
                            b, :, points_list[b][n][1], points_list[b][n][0] : points_list[b][n][2] + 1
                        ].sum()
                        == (points_list[b][n][2] - points_list[b][n][0] + 1) * 3
                    )

    @pytest.mark.parametrize('color', (torch.tensor([0.5, 0.3, 0.15]), torch.tensor([0.23, 0.33, 0.8])))
    def test_color_background(self, device, color):
        image = torch.zeros(1, 3, 40, 40, device=device)
        image[:, 0, :, :] = color[0]
        image[:, 1, :, :] = color[1]
        image[:, 2, :, :] = color[2]
        image_w_rectangle = image.clone()
        p1 = (1, 5)
        p2 = (30, 39)
        points = torch.tensor([[[p1[1], p1[0], p2[1], p2[0]]]], device=device)

        draw_rectangle(image_w_rectangle, points, color=torch.tensor([1.0]))
        assert (
            torch.abs(
                (image_w_rectangle - image).sum()
                - (1 - color[0]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
                - (1 - color[1]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
                - (1 - color[2]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1] + 1) - 4)
            )
            <= 0.0001
        )

    @pytest.mark.parametrize('color', (torch.tensor([0.34, 0.63, 0.16]), torch.tensor([0.29, 0.13, 0.48])))
    def test_color_foreground(self, device, color):
        image = torch.zeros(1, 3, 50, 40, device=device)
        image_w_rectangle = image.clone()
        p1 = (10, 4)
        p2 = (11, 40)
        points = torch.tensor([[[p1[1], p1[0], p2[1], p2[0]]]], device=device)

        draw_rectangle(image_w_rectangle, points, color=color)

        # corners are double counted, no plus 1 for y since p2[1] of 40 already lies outside of the image
        assert (
            torch.abs(
                (image_w_rectangle - image).sum()
                - (color[0]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
                - (color[1]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
                - (color[2]) * (2 * (p2[0] - p1[0] + 1) + 2 * (p2[1] - p1[1]) - 4)
            )
            <= 0.0001
        )
