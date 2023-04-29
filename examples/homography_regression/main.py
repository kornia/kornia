import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import kornia as dgm


def load_homography(file_name):
    """Load a homography from text file."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")
    return torch.from_numpy(np.loadtxt(file_name)).float()


def load_image(file_name):
    """Load the image with OpenCV and converts to torch.Tensor."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")

    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # convert image to torch tensor
    tensor = dgm.utils.image_to_tensor(img).float() / 255.0
    tensor = tensor.view(1, *tensor.shape)  # 1xCxHxW

    return tensor, img


class MyHomography(nn.Module):
    def __init__(self):
        super().__init__()
        self.homo = nn.Parameter(torch.Tensor(3, 3))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.homo)

    def forward(self):
        return torch.unsqueeze(self.homo, dim=0)  # 1x3x3


def HomographyRegressionApp():
    # Training settings
    parser = argparse.ArgumentParser(description='Homography Regression with photometric loss.')
    parser.add_argument('--input-dir', type=str, required=True, help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True, help='the path to output the results.')
    parser.add_argument(
        '--num-iterations', type=int, default=1000, metavar='N', help='number of training iterations (default: 1000)'
    )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S', help='random seed (default: 666)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--log-interval-vis',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before visual logging training status',
    )
    args = parser.parse_args()

    # define the device to use for inference
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    # load the data
    img_src, _ = load_image(os.path.join(args.input_dir, 'img1.ppm'))
    img_dst, _ = load_image(os.path.join(args.input_dir, 'img2.ppm'))

    # instantiate the homography warper from `kornia`
    height, width = img_src.shape[-2:]
    warper = dgm.HomographyWarper(height, width)

    # create the homography as the parameter to be optimized
    dst_homo_src = MyHomography().to(device)

    # create optimizer
    optimizer = optim.Adam(dst_homo_src.parameters(), lr=args.lr)

    # main training loop

    for iter_idx in range(args.num_iterations):
        # send data to device
        img_src, img_dst = img_src.to(device), img_dst.to(device)

        # warp the reference image to the destiny with current homography
        img_src_to_dst = warper(img_src, dst_homo_src())

        # compute the photometric loss
        loss = F.l1_loss(img_src_to_dst, img_dst, reduction='none')

        # propagate the error just for a fixed window
        w_size = 100  # window size
        h_2, w_2 = height // 2, width // 2
        loss = loss[..., h_2 - w_size : h_2 + w_size, w_2 - w_size : w_2 + w_size]
        loss = torch.mean(loss)

        # compute gradient and update optimizer parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % args.log_interval == 0:
            print(f'Train iteration: {iter_idx}/{args.num_iterations}\tLoss: {loss.item():.6}')
            print(dst_homo_src.homo)

        def draw_rectangle(image, dst_homo_src):
            height, width = image.shape[:2]
            pts_src = torch.FloatTensor(
                [[[-1, -1], [1, -1], [1, 1], [-1, 1]]]  # top-left  # bottom-left  # bottom-right  # top-right
            ).to(dst_homo_src.device)
            # transform points
            pts_dst = dgm.transform_points(torch.inverse(dst_homo_src), pts_src)

            def compute_factor(size):
                return 1.0 * size / 2

            def convert_coordinates_to_pixel(coordinates, factor):
                return factor * (coordinates + 1.0)

            # compute conversion factor
            x_factor = compute_factor(width - 1)
            y_factor = compute_factor(height - 1)
            pts_dst = pts_dst.cpu().squeeze().detach().numpy()
            pts_dst[..., 0] = convert_coordinates_to_pixel(pts_dst[..., 0], x_factor)
            pts_dst[..., 1] = convert_coordinates_to_pixel(pts_dst[..., 1], y_factor)

            # do the actual drawing
            for i in range(4):
                pt_i, pt_ii = tuple(pts_dst[i % 4]), tuple(pts_dst[(i + 1) % 4])
                image = cv2.line(image, pt_i, pt_ii, (255, 0, 0), 3)
            return image

        if iter_idx % args.log_interval_vis == 0:
            # merge warped and target image for visualization
            img_src_to_dst = warper(img_src, dst_homo_src())
            img_vis = 255.0 * 0.5 * (img_src_to_dst + img_dst)
            img_vis_np = dgm.utils.tensor_to_image(img_vis)
            image_draw = draw_rectangle(img_vis_np, dst_homo_src())
            # save warped image to disk
            file_name = os.path.join(args.output_dir, f'warped_{iter_idx}.png')
            cv2.imwrite(file_name, image_draw)


if __name__ == "__main__":
    HomographyRegressionApp()
