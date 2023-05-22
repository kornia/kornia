import math

import pytest
import torch

from kornia.testing import assert_close
from kornia.utils import create_meshgrid, draw_convex_polygon, draw_rectangle
from kornia.utils.draw import draw_line, draw_point2d


class TestDrawPoint:
    """Test drawing individual pixels."""

    def test_draw_point2d_rgb(self, dtype, device):
        """Test plotting multiple [x, y] points."""
        points = [(1, 3), (2, 4)]
        color = torch.tensor([5, 10, 15])
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        draw_point2d(img, points, color)
        for x, y in points:
            assert_close(img[:, x, y], color.to(img.dtype))

    def test_draw_point2d_grayscale_third_order(self, dtype, device):
        """Test plotting multiple [x, y] points on a (1, m, n) image."""
        points = [(1, 3), (2, 4)]
        color = torch.tensor([100])
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        draw_point2d(img, points, color)
        for x, y in points:
            assert_close(img[:, x, y], color.to(img.dtype))

    def test_draw_point2d_grayscale_second_order(self, dtype, device):
        """Test plotting multiple [x, y] points on a (m, n) image."""
        points = [(1, 3), (2, 4)]
        color = torch.tensor([100])
        img = torch.zeros(8, 8, dtype=dtype, device=device)
        draw_point2d(img, points, color)
        for x, y in points:
            assert_close(torch.unsqueeze(img[x, y], dim=0), color.to(img.dtype))

    def test_draw_point2d_with_mismatched_dims(self, dtype, device):
        """Test that we raise if the len of the color tensor != the # of image channels."""
        points = [(1, 3), (2, 4)]
        color = torch.tensor([100])
        img = torch.zeros(3, 8, 8, dtype=dtype, device=device)
        with pytest.raises(Exception):
            draw_point2d(img, points, color)


class TestDrawLine:
    def test_draw_line_vertical(self, dtype, device):
        """Test drawing a vertical line."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([6, 2]), torch.tensor([6, 0]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    def test_draw_line_horizontal(self, dtype, device):
        """Test drawing a horizontal line."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([6, 4]), torch.tensor([0, 4]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    def test_draw_line_with_big_coordinates(self, dtype, device):
        """Test drawing a line with big coordinates."""
        img = torch.zeros(1, 500, 500, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([200, 200]), torch.tensor([400, 200]), torch.tensor([255]))
        img_mask = torch.zeros(1, 500, 500, dtype=dtype, device=device)
        img_mask[:, 200, 200:401] = 255
        assert_close(img, img_mask)

    def test_draw_line_m_lte_neg1(self, dtype, device):
        """Test drawing a line with m <= -1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([0, 7]), torch.tensor([6, 0]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    def test_draw_line_m_lt_0_gte_neg1(self, dtype, device):
        """Test drawing a line with -1 < m < 0."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([1, 5]), torch.tensor([7, 0]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    def test_draw_line_m_gt_0_lt_1(self, dtype, device):
        """Test drawing a line with 0 < m < 1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([0, 0]), torch.tensor([6, 2]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [255.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 255.0, 255.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    def test_draw_line_m_gte_1(self, dtype, device):
        """Test drawing a line with m >= 1."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        img = draw_line(img, torch.tensor([3, 7]), torch.tensor([1, 4]), torch.tensor([255]))
        img_mask = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        assert_close(img, img_mask)

    @pytest.mark.parametrize(
        'p1', [torch.tensor([-1, 0]), torch.tensor([0, -1]), torch.tensor([8, 0]), torch.tensor([0, 8])]
    )
    def test_p1_out_of_bounds(self, p1, dtype, device):
        """Tests that an exception is raised if p1 is out of bounds."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, p1, torch.tensor([0, 0]), torch.tensor([255]))

        assert 'p1 is out of bounds.' == str(excinfo.value)

    @pytest.mark.parametrize(
        'p2', [torch.tensor([-1, 0]), torch.tensor([0, -1]), torch.tensor([8, 0]), torch.tensor([0, 8])]
    )
    def test_p2_out_of_bounds(self, p2, dtype, device):
        """Tests that an exception is raised if p2 is out of bounds."""
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), p2, torch.tensor([255]))

        assert 'p2 is out of bounds.' == str(excinfo.value)

    @pytest.mark.parametrize('img_size', [(200, 100), (32, 3, 20, 20)])
    def test_image_size(self, img_size, dtype, device):
        img = torch.zeros(*img_size, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([255]))

        assert 'image must have 3 dimensions (C,H,W).' == str(excinfo.value)

    @pytest.mark.parametrize('img_size,color', [((1, 8, 8), torch.tensor([23, 53])), ((3, 8, 8), torch.tensor([255]))])
    def test_color_image_channel_size(self, img_size, color, dtype, device):
        img = torch.zeros(*img_size, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, torch.tensor([0, 0]), torch.tensor([1, 1]), color)

        assert 'color must have the same number of channels as the image.' == str(excinfo.value)

    @pytest.mark.parametrize(
        'p1,p2',
        [
            ((0, 1), (1, 5, 2)),
            ((0, 1, 2), (1, 5)),
            (torch.tensor([0, 1]), torch.tensor([0, 2, 3])),
            (torch.tensor([0, 1, 5]), torch.tensor([0, 2])),
        ],
    )
    def test_point_size(self, p1, p2, dtype, device):
        img = torch.zeros(1, 8, 8, dtype=dtype, device=device)
        with pytest.raises(ValueError) as excinfo:
            draw_line(img, p1, p2, torch.tensor([255]))

        assert 'p1 and p2 must have length 2.' == str(excinfo.value)


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

        points = torch.tensor(points_list, device=device)

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


class TestFillConvexPolygon:
    def test_circle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        n = 5000
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)[None].expand(b, n)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        x = (2 * math.pi * t).cos()
        y = (2 * math.pi * t).sin()
        ctr = 200
        radius = 200
        pts = ctr + radius * torch.stack((x, y), dim=-1)
        poly_im = draw_convex_polygon(im, pts, color)
        XY = create_meshgrid(h, w, normalized_coordinates=False, device=device, dtype=dtype)
        inside = (((XY[..., 1] - ctr) ** 2 + (XY[..., 0] - ctr) ** 2).sqrt() <= radius)[:, None].expand(b, c, h, w)
        circ_im = inside * color[..., None, None]
        assert (circ_im - poly_im).abs().mean() <= 1e-4

    def test_ellipse(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        n = 5000
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)[None].expand(b, n)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        lam = 2
        x = lam * (2 * math.pi * t).cos()
        y = (2 * math.pi * t).sin()
        ctr = 200
        radius = 100
        pts = ctr + radius * torch.stack((x, y), dim=-1)
        poly_im = draw_convex_polygon(im, pts, color)
        XY = create_meshgrid(h, w, normalized_coordinates=False, device=device, dtype=dtype)
        inside = (((XY[..., 1] - ctr) ** 2 + ((XY[..., 0] - ctr) / lam) ** 2).sqrt() <= radius)[:, None].expand(
            b, c, h, w
        )
        ellipse_im = inside * color[..., None, None]
        assert (ellipse_im - poly_im).abs().mean() <= 1e-4

    def test_rectangle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        pts = torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        assert_close(rect_im, poly_im)

    def test_batch(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = torch.tensor(
            [[[4, 4], [12, 4], [12, 8], [4, 8]], [[0, 0], [4, 0], [4, 4], [0, 4]]], dtype=dtype, device=device
        )
        color = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        assert_close(rect_im, poly_im)

    def test_batch_variable_size(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = [
            torch.tensor([[4, 4], [12, 4], [12, 8], [4, 8]], dtype=dtype, device=device),
            torch.tensor([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]], dtype=dtype, device=device),
        ]
        color = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        assert_close(rect_im, poly_im)

    def test_batch_color_no_batch(self, device, dtype):
        im = torch.rand(2, 3, 12, 16, dtype=dtype, device=device)
        pts = [
            torch.tensor([[4, 4], [12, 4], [12, 8], [4, 8]], dtype=dtype, device=device),
            torch.tensor([[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]], dtype=dtype, device=device),
        ]
        color = torch.tensor([0.5, 0.5, 0.75], dtype=dtype, device=device)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.tensor([[[4, 4, 12, 8]], [[0, 0, 4, 4]]], dtype=dtype, device=device)
        rect_im = draw_rectangle(im.clone(), rect, color, fill=True)
        assert_close(rect_im, poly_im)

    def test_out_of_bounds_rectangle(self, device, dtype):
        b, c, h, w = 1, 3, 500, 500
        im = torch.zeros(b, c, h, w, device=device, dtype=dtype)
        color = torch.tensor([1, 1, 1], device=device, dtype=dtype)[None].expand(b, c)
        pts = 350 + torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        assert_close(rect_im, poly_im)
        pts = -150 + torch.tensor([[[50, 50], [200, 50], [200, 250], [50, 250]]], device=device, dtype=dtype)
        poly_im = draw_convex_polygon(im.clone(), pts, color)
        rect = torch.cat((pts[..., 0, :], pts[..., 2, :]), dim=-1)[:, None]
        rect_im = draw_rectangle(im.clone(), rect, color[:, None], fill=True)
        assert_close(rect_im, poly_im)
